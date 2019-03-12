import sys
import os.path
import math
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from create_answer_map_test import data_loader
import config
import model_showask as model
import utils
import numpy
import numpy as np

idmap = {0: 'cylinder',
 1: 'yellow',
 2: 'sphere',
 3: 'blue',
 4: 'rubber',
 5: 'purple',
 6: '1',
 7: '0',
 8: '3',
 9: '2',
 10: '5',
 11: '4',
 12: '7',
 13: '6',
 14: '8',
 15: 'True',
 16: 'red',
 17: 'brown',
 18: 'cube',
 19: 'cyan',
 20: 'gray',
 21: 'False',
 22: 'metal',
 23: 'large',
 24: 'green',
 25: 'small'}

output_list = []

def run(net, train=False, prefix='', epoch=0):

	net.eval()
	answ = []
	idxs = []
	accs = []
	tq = tqdm(data_loader(prefix), desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
	for v, q, qid, idx, q_len in tq:
		var_params = {
			'volatile': not train,
			'requires_grad': False,
		}
		print('HOLA HOLA')
		# print(np.asarray(v).shape)
		v = Variable(torch.Tensor(v).cuda(async=True), **var_params)
		q = Variable(torch.LongTensor(q).cuda(async=True), **var_params)
		q_len = Variable(torch.Tensor(q_len).cuda(async=True), **var_params)
		idx = Variable(torch.LongTensor(idx).cuda(async=True), **var_params)
		out = net(v, q, q_len)
		_, predicted_index = out.max(dim=1, keepdim=True)
		predicted_val = idmap[int(predicted_index.data.cpu().numpy()[0])]
		output_list.append((qid[0], predicted_val))
		

def main():
	cudnn.benchmark = True
	net = nn.DataParallel(model.Net(87)).cuda()
	net.load_state_dict(torch.load('logs/final.pth')['weights'])
	_ = run(net, train=False, prefix='train', epoch=0)
	import csv
	with open("solution.csv", "wb") as f:
		writer = csv.writer(f)
		writer.writerows(output_list)
if __name__ == '__main__':
	main()