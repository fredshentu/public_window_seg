require 'torch'
require 'cutorch'
require 'image'

npy4th = require 'npy4th'
local coco = require 'coco'
local maskApi = coco.MaskApi
local cmd = torch.CmdLine()
cmd:option('-img','./img1origin.jpg' ,'path/to/test/image')
cmd:option('-masks', '././deepmask/img1masks.jpg.npy', "path to plot masks")
local config = cmd:parse(arg)

local img = image.load(config.img)
-- load pre-computed python mask
local masks = npy4th.loadnpy(config.masks)
print(torch.max(masks))
print(#masks)
-- exit()
local res = img:clone()
maskApi.drawMasks(res, masks, 30)
image.save("./test.jpg",res)
print('| done')
collectgarbage()
