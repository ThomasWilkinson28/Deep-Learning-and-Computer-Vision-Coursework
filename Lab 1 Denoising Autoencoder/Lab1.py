import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from model import autoencoderMLP4Layer
import argparse

def main():
    #Read arguments from command line
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-l', metavar='state', type=str, help='parameter file (.pth)')
   
    args = argParser.parse_args()

    save_file = None
    if args.l != None:
        save_file = args.l


    #Define transformation which will be used to convert images to tensors
    train_transform = transforms.Compose([transforms.ToTensor()])
    #Load dataset
    train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)

    #Optionality for cpu or gpu
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('\t\tusing device ', device)

    #Configure model
    model = autoencoderMLP4Layer(N_input=28*28, N_bottleneck= 8, N_output=28*28)
    model.load_state_dict(torch.load(save_file))
    model.to(device)
    model.eval()

    idx = 0
    while idx >= 0:
        #Prompt user for an index
        idx = int(input("Enter an index between 0 and 59999: "))
        if 0 <= idx <= train_set.data.size()[0]:
            img = train_set.data[idx]
            img = img.type(torch.float32)
            img = (img - torch.min(img)) / torch.max(img)
            test_encoder(img, model)
            image_denoising(img, model)
            interpolate(img, train_set, model)


def test_encoder(img, model):
    img = img.view(1, img.shape[0]*img.shape[1]).type(torch.FloatTensor)
    with torch.no_grad():
        output = model(img)

    output = output.view(28, 28).type(torch.FloatTensor)
    img = img.view(28, 28).type(torch.FloatTensor)
    f = plt.figure()
    f.add_subplot(1,2,1)
    plt.imshow(img, cmap='gray')
    f.add_subplot(1,2,2)
    plt.imshow(output, cmap='gray')
    plt.show()


def image_denoising(img, model):
    #Add noise using torch.rand()
    noise = torch.rand(28, 28)
    noisy_img = img + noise*0.5
    noisy_img = noisy_img.view(1, noisy_img.shape[0]*noisy_img.shape[1]).type(torch.FloatTensor)
    with torch.no_grad():
        output = model(noisy_img)

    output = output.view(28, 28).type(torch.FloatTensor)
    noisy_img = noisy_img.view(28, 28).type(torch.FloatTensor)
    f = plt.figure()
    f.add_subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    f.add_subplot(1,3,2)
    plt.imshow(noisy_img, cmap='gray')
    f.add_subplot(1,3,3)
    plt.imshow(output, cmap='gray')
    plt.show()


def interpolate(img1, train_set, model):
    steps = 8
    idx = int(input("Enter second image index: "))
    if 0 <= idx <= train_set.data.size()[0]:
        img2 = train_set.data[idx].type(torch.float32)
        img2 = (img2 - torch.min(img2)) / torch.max(img2)
        
        img1 = img1.view(1, img1.shape[0]*img1.shape[1]).type(torch.FloatTensor)
        img2 = img2.view(1, img2.shape[0]*img2.shape[1]).type(torch.FloatTensor)
        
        with torch.no_grad():
            img1_encoded = model.encode(img1)
            img2_encoded = model.encode(img2)

        amount = torch.linspace(0, 1, steps)
        bottlenecks = [(1-x)*img1_encoded + x*img2_encoded for x in amount]
        outputs = [model.decode(b).view(28, 28).detach().cpu().numpy() for b in bottlenecks]

        plt.figure(figsize=(12,2))
        for i, img in enumerate(outputs):
            plt.subplot(1, steps, i+1)
            plt.imshow(img, cmap='gray')
        plt.suptitle(f"Interpolation between first and second images chosen")
        plt.show()

        
###################################################################

if __name__ == '__main__':
    main()

