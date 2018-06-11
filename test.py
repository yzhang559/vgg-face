import collections, torch, torchvision, numpy, h5py
import Image
from torchvision import transforms, utils
from torch.autograd import Variable
from model import vgg13


imsize = 224
loader = transforms.Compose([transforms.Scale(imsize), transforms.ToTensor()])
def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU


model = vgg13()

model.features = torch.nn.Sequential(collections.OrderedDict(zip(['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4', 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5'], model.features)))
model.classifier = torch.nn.Sequential(collections.OrderedDict(zip(['fc6', 'relu6', 'drop6', 'fc7', 'relu7', 'drop7', 'fc8', 'prob'], model.classifier)))

state_dict = h5py.File('vggface.h5', 'r') 
torch.load('VGG_FACE.caffemodel.pth')
model.load_state_dict({l : torch.from_numpy(numpy.array(v)).view_as(p) for k, v in state_dict.items() for l, p in model.named_parameters() if k in l})
model.cuda() #assumes that you're using GPU
model.eval()


image = image_loader('ak.png')
prob = model(image).cpu().detach().numpy()
idx = numpy.argmax(prob, axis=1)
max_prob = prob[0, idx]

with open("names.txt") as fp:
    for i, line in enumerate(fp):
        if i == idx:
            name = line

print str(name) + 'Max_prob: ' + str(max_prob)