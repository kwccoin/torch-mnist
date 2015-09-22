require "nn"

local function train(model, criterion, opt, optimState, train_x, train_y, confusion, logger)
  local time = sys.clock()
  --local logger = optim.Logger(paths.concat(opt.save,'train.log'))
  local shuffle = torch.randperm(train_x:size(1))
  model:training()
  local parameters,gradParameters = model:getParameters()

  confusion:zero()

  for t = 1,train_x:size(1),opt.batchSize do
    if t + opt.batchSize-1 > train_x:size(1) then
      break
    end

    xlua.progress(t, train_x:size(1))

    -- create mini batch
    local inputs = torch.Tensor(opt.batchSize, 1, train_x:size(2), train_x:size(3))
    local targets = torch.Tensor(opt.batchSize)
    for i = t, t+opt.batchSize-1 do
      inputs[i-t+1]:copy(train_x[shuffle[i]])
      targets[i-t+1] = train_y[shuffle[i]]+1  -- target is 1 ~ 10
    end
    if opt.type == 'cuda' then
      inputs = inputs:cuda()
      targets = targets:cuda()
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
      for i = 1,opt.batchSize do
        confusion:add(outputs[i], targets[i])
      end

      -- normalize gradients and f(X)
      -- gradParameters:div(inputs:size(1))
      f = f/inputs:size(1)

      -- return f and df/dX
      return f,gradParameters
    end

    -- optimize on current mini-batch
    optim.sgd(feval, parameters, optimState)

    -- print confusion matrix
  end

  print(confusion)
  logger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  confusion:zero()
end

return train
