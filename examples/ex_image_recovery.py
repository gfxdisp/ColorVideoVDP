import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import ex_utils as utils
import pycvvdp
import time

from torchvision.transforms import GaussianBlur

debug = False

class ImageRecovery(torch.nn.Module):
    def __init__(self, ref_img, initialization="random"):
        super().__init__()        
        if initialization == "random":
            self.rec_image = torch.nn.Parameter( torch.rand_like( ref_img ) )
        elif initialization == "blurred":
            sigma = 4
            gb = GaussianBlur(int(sigma*4)+1, sigma)
            self.rec_image = torch.nn.Parameter(gb.forward(ref_img))
        else:
            assert False
        
    def forward(self):
        return self.rec_image


device = torch.device("cuda:0")

I_ref = pycvvdp.load_image_as_array(os.path.join('example_media', 'wavy_facade.png'))
T_ref = torch.as_tensor( I_ref.astype(np.float32) ).to(device).permute((2,0,1))/(2**16-1)


model = ImageRecovery( T_ref, initialization="blurred" )
model.to(device)

#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0, weight_decay=0, dampening=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

cvvdp = pycvvdp.cvvdp(display_name='standard_4k')
#cvvdp.masking_model = 'mult-none'
    
#loss_fn = torch.nn.MSELoss( reduction='sum' )

loss_fn = lambda pred, y : cvvdp.loss( pred, y, dim_order="CHW")
# + 0.1*torch.sum((pred - y)**2)
#cvvdp.loss( pred, y, dim_order="CHW") + 10 *
# torch.sum((pred - y)**2) + 0.1 * 

plt.ion()
figure, ax = plt.subplots(figsize=(10, 8))

for kk in range(1000):
    print( f"Iteration {kk}" )
    optimizer.zero_grad()
    pred = model().clamp(0.,1.)
    loss = loss_fn(pred, T_ref)

    if kk % 20 == 0:
        ax.imshow( pred.detach().permute((1,2,0)).cpu().numpy() )
        ax.set_title( f"Iteration {kk}: loss {loss.item()}" )
        figure.canvas.draw()
        figure.canvas.flush_events()
 
    #time.sleep(0.1)

    # Backpropagation
    loss.backward()
    optimizer.step()

plt.waitforbuttonpress()