require "qt"
require "qtgui"
require "qtwidget"
require "qtuiloader"
require "qttorch"
require "xlua"

require "torch"
require "nn"

function pred(model, input)
	local output = model:forward(input)
	-- model use LogSoftMax()
	return torch.exp(output)
end

function main()
  torch.setdefaulttensortype('torch.FloatTensor')

	local w = qtuiloader.load("mnist_gui.ui")
	local listener = qt.QtLuaListener(w)
	local painter = qt.QtLuaPainter(w.frame)
	local dragging = false
	local button = ""
	local frame_pos = w.frame.pos:totable()
	local frame_size = {width = w.frame.width, height = w.frame.height}
	local scale = 5
	local input = torch.Tensor(28,28):fill(0)

	local model = torch.load("./float_results/model.net")
	model:evaluate()

	local function frame_coord(x_, y_)
		-- it doesn't seem to work... ???
		-- local newpos = w.frame:mapFromGlobal(qt.QPoint{x=x,y=y}):totable()
		local newpos = {x = x_ - frame_pos.x, y = y_ - frame_pos.y}
		return newpos
	end

	local function check(pos)
		if pos.x >= 0 and pos.y >= 0 and
			 pos.x < frame_size.width and
			 pos.y < frame_size.height then
			return true
		end
		return false
	end

	local function update_bar()
		output = pred(model,input)
		w.bar0.value  = math.floor(output[1]*100)
		w.bar1.value  = math.floor(output[2]*100)
		w.bar2.value  = math.floor(output[3]*100)
		w.bar3.value  = math.floor(output[4]*100)
		w.bar4.value  = math.floor(output[5]*100)
		w.bar5.value  = math.floor(output[6]*100)
		w.bar6.value  = math.floor(output[7]*100)
		w.bar7.value  = math.floor(output[8]*100)
		w.bar8.value  = math.floor(output[9]*100)
		w.bar9.value  = math.floor(output[10]*100)
	end

	-- mouse press
	qt.connect(listener,
		"sigMousePress(int,int,QByteArray,QByteArray,QByteArray)",
		function(x_,y_,name)
			local pos = frame_coord(x_,y_)
			if check(pos) then
				-- print("drag on")
				dragging = true
				button = name
			end
		end)

	-- mouse release
	qt.connect(listener,
		"sigMouseRelease(int,int,QByteArray,QByteArray,QByteArray)",
		function()
			if dragging == true then
				-- print("drag off")
				dragging = false
			end
		end)

	-- mouse move
	qt.connect(listener,
		"sigMouseMove(int,int,QByteArray,QByteArray)",
		function(x_,y_,name)
			local pos = frame_coord(x_,y_)
			if dragging == true and check(pos) then
				-- print("move:",pos.x,pos.y)
				local px = math.floor(pos.x / scale)
				local py = math.floor(pos.y / scale)
				if button == "RightButton" then
					painter:setcolor("white")
					input[py+1][px+1] = 0
				else
					painter:setcolor("black")
					input[py+1][px+1] = 1
				end
				painter:rectangle(px*scale,py*scale,scale,scale)
				painter:fill()
				update_bar()
			end
		end)

	-- reset
	qt.connect(w.resetButton, "clicked()",
		function()
			painter:showpage()
			input:fill(0)
		  update_bar()
		end)

  update_bar()
	painter:showpage()
	w:show()
end


main()
