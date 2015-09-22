
require "nn"

local ninputs = 28*28
local nhiddens = 100
local noutputs = 10

local model = nn.Sequential()
model:add(nn.Reshape(ninputs))
model:add(nn.Linear(ninputs,nhiddens))
model:add(nn.Sigmoid())
model:add(nn.Linear(nhiddens,noutputs))

-- criterion
model:add(nn.LogSoftMax())
local criterion = nn.ClassNLLCriterion()

return {model = model, criterion = criterion}
