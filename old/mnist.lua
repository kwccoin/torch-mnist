-- MLP for MNIST Classification

require 'nn'
require 'optim'
require 'pl'
require 'torch'
require 'xlua'

function init(nthreads,seed)
  torch.setdefaulttensortype('torch.FloatTensor')
  torch.setnumthreads(nthreads)
  torch.manualSeed(seed)
end

-- data
function load_mnist()
  local mnist = require 'mnist'
  local trainset = mnist.traindataset()
  local testset = mnist.testdataset()
  local train_x, train_y = trainset.data:float()/255, trainset.label
  local test_x, test_y = testset.data:float()/255, testset.label
  return train_x, train_y, test_x, test_y
end

function build_mlp(ninputs,nhiddens,noutputs)
  -- layer
  local model = nn.Sequential()
  model:add(nn.Reshape(ninputs))
  model:add(nn.Linear(ninputs,nhiddens))
  model:add(nn.Sigmoid())
  model:add(nn.Linear(nhiddens,noutputs))

  return model
end

function set_criterion(model)
  -- cross entropy
  model:add(nn.LogSoftMax())
  local criterion = nn.ClassNLLCriterion()

  return criterion
end

--function train(dataset)
function train(model, criterion, batchSize, optimState, train_x, train_y, confusion, logger)
  local time = sys.clock()
  local shuffle = torch.randperm(train_x:size(1))
  model:training()
  local parameters,gradParameters = model:getParameters()

  confusion:zero()

  for t = 1,train_x:size(1),batchSize do
    if t + batchSize-1 > train_x:size(1) then
      break
    end

    xlua.progress(t, train_x:size(1))

    -- create mini batch
    local inputs = torch.Tensor(batchSize, 1, train_x:size(2), train_x:size(3))
    local targets = torch.Tensor(batchSize)
    for i = t, t+batchSize-1 do
      inputs[i-t+1]:copy(train_x[shuffle[i]])
      targets[i-t+1] = train_y[shuffle[i]]+1  -- target is 1 ~ 10
    end

    local feval = function(x)
      if x ~= parameters then
        parameters:copy(x)
      end
      gradParameters:zero()

      -- fprop & bprop
      local outputs = model:forward(inputs)
      local f = criterion:forward(outputs, targets)
      local df_do = criterion:backward(outputs, targets)
      model:backward(inputs, df_do)

      -- update cnfusion matrix;
      for i = 1,batchSize do
        confusion:add(outputs[i], targets[i])
      end

      -- normalize gradients and f(X)
      gradParameters:div(inputs:size(1))
      f = f/inputs:size(1)

      -- return f and df/dX
      return f,gradParameters
    end

    -- optimize on current mini-batch
    optim.sgd(feval, parameters, optimState)

    -- print confusion matrix
    logger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  end

  print(confusion)
  confusion:zero()
end

function test(model, batchSize, test_x, test_y, confusion)
  local time = sys.clock()
  model:evaluate()
  confusion:zero()

  for t = 1,test_x:size(1),batchSize do
    if t + batchSize-1 > test_x:size(1) then
      break
    end

    xlua.progress(t, test_x:size(1))

    -- create mini batch
    local inputs = torch.Tensor(batchSize, 1, 28, 28)
    local targets = torch.Tensor(batchSize)
    for i = t, t+batchSize-1 do
      inputs[i-t+1]:copy(test_x[i])
      targets[i-t+1] = test_y[i]+1
    end

    local outputs = model:forward(inputs)
      -- update cnfusion matrix;
    for i = 1,batchSize do
        confusion:add(outputs[i], targets[i])
    end
  end

  print(confusion)
  confusion:zero()
end


function main()
  local ninputs = 28*28
  local nhiddens = 100
  local noutputs = 10
  local nbatches = 100
  local nthreads = 12
  local seed = 1

  init(nthreads,seed)

  -- training
  local optimState = {
    learningRate = 1.0,
    weightDecay = 0.00001,
    momentum = 0.1,
    learningRateDecay = 0
  }
  
  -- logger
  local train_logger = optim.Logger('train.log')
  local test_logger = optim.Logger('test.log')
  
  local train_x, train_y, test_x, test_y = load_mnist()
  local classes = {0,1,2,3,4,5,6,7,8,9}
  local confusion = optim.ConfusionMatrix(classes)
  local model = build_mlp(ninputs,nhiddens,noutputs)
  local criterion = set_criterion(model)

  print("num of threads : " .. torch.getnumthreads())
  epoch = 1
  for epoch=1, 50 do
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [nbatches = ' .. nbatches .. ']')
    train(model,criterion,nbatches,optimState,train_x,train_y,confusion,train_logger)
    print("==> test")
    test(model,nbatches,test_x,test_y,confusion)
  end
end

main()
