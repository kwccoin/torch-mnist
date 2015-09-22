require "nn"

local function test(model, opt, test_x, test_y, confusion, logger)
  local time = sys.clock()
  --local logger = optim.Logger(paths.concat(opt.save,'test.log'))
  model:evaluate()
  confusion:zero()

  for t = 1,test_x:size(1),opt.batchSize do
    if t + opt.batchSize-1 > test_x:size(1) then
      break
    end

    xlua.progress(t, test_x:size(1))

    -- create mini batch
    local inputs = torch.Tensor(opt.batchSize, 1, 28, 28)
    local targets = torch.Tensor(opt.batchSize)
    for i = t, t+opt.batchSize-1 do
      inputs[i-t+1]:copy(test_x[i])
      targets[i-t+1] = test_y[i]+1
    end
    if opt.type == 'cuda' then
      inputs = inputs:cuda()
      targets = targets:cuda()
    end

    local outputs = model:forward(inputs)
      -- update cnfusion matrix;
    for i = 1,opt.batchSize do
        confusion:add(outputs[i], targets[i])
    end
  end

  print(confusion)
  logger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
  if opt.maxValid < confusion.totalValid then
    opt.maxValid = confusion.totalValid
    -- save model
    local f = paths.concat(opt.save,"model.net")
    print(sys.COLORS.blue .. "==> saving model to " .. f)
    torch.save(f,model)
  end
  confusion:zero()
end

return test
