require 'nn'
require 'optim'
require "torch"
require 'xlua'

-- MNIST classifier

function init(nthreads,seed)
  print("==> num of threads: " .. nthreads)
  print("==> random seed: " .. seed)
  torch.setdefaulttensortype('torch.FloatTensor')
  torch.setnumthreads(nthreads)
  torch.manualSeed(seed)
end

function load_mnist()
  local mnist = require 'mnist'
  local trainset = mnist.traindataset()
  local testset = mnist.testdataset()
  local train_x, train_y = trainset.data:float()/255, trainset.label
  local test_x, test_y = testset.data:float()/255, testset.label
  return train_x, train_y, test_x, test_y
end

function main()
  local lapp = require 'pl.lapp'
  local opt = lapp[[
    -r,--learningRate        (default 0.5)         learning rate
    -d,--learningRateDecay   (default 1e-7)        learning rate decay (in # samples)
    -w,--weightDecay         (default 1e-5)        weight decay
    -m,--momentum            (default 0.1)         momentum
    -t,--type                (default float)       float or cuda
    -i,--devid               (default 1)           device ID (if using CUDA)
    -n,--threads             (default 8)           number of threads
    -b,--batchSize           (default 128)         batch size
    -e,--epoch               (default 50)          max epoch (-1 is infinity)
    -l,--load                (default none)        load model from specified file
    -s,--seed                (default 1)           seed of random
    -o,--save                (default results)     save directory
]]

  -- this is used for model save
  opt.maxValid = 0

  init(opt.threads, opt.seed)

  local optimState = {
    learningRate = opt.learningRate,
    weightDecay = opt.weightDecay,
    momentum = opt.momentum,
    learningRateDecay = opt.learningRateDecay
  }

  local t = require "model"
  local model = nil
  if opt.load ~= "none" then
    model = torch.load(opt.load)
  else
    model = t.model
  end
  local criterion = t.criterion
  local train = require "train"
  local test = require "test"

  local train_x, train_y, test_x, test_y = load_mnist()
  local classes = {0,1,2,3,4,5,6,7,8,9}
  local confusion = optim.ConfusionMatrix(classes)

  local train_logger = optim.Logger(paths.concat(opt.save,'train.log'))
  local test_logger = optim.Logger(paths.concat(opt.save,'test.log'))

  if opt.type == 'cuda' then
     require 'cutorch'
     require 'cunn'
     print(sys.COLORS.red ..  '==> switching to CUDA')
     cutorch.setDevice(opt.devid)
     local devid = cutorch.getDevice()
     local gpu_info = cutorch.getDeviceProperties(devid)
     print(sys.COLORS.red ..  '==> using GPU #' .. devid .. ' ' .. gpu_info.name .. ' (' .. math.ceil(gpu_info.totalGlobalMem/1024/1024/1024) ..  'GB)')
     model:cuda()
     criterion:cuda()
  end

  epoch = 1
  while true do
    print('==> doing epoch on training data:')
    print("==> online epoch # " .. epoch .. ' [batch size = ' .. opt.batchSize .. ']')
    train(model,criterion,opt,optimState,train_x,train_y,confusion,train_logger)
    print("==> test")
    test(model,opt,test_x,test_y,confusion,test_logger)

    epoch = epoch + 1
    if opt.epoch > 0 and opt.epoch < epoch then
      break
    end
  end

end

main()
