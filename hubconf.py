import torch
from main import VPRModel  # Replace with your model's class

def MixVPR(pretrained=True, **kwargs):
    """
    Loads the model. If `pretrained` is True, it loads pre-trained weights.
    """
    model = VPRModel(**kwargs)  # Replace with your model initialization
    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            '/home/jarvis/jw_ws/release/MixVPR/model.ckpt',  # Update with your weights URL
            map_location=torch.device('cpu')  # Change if GPU loading is needed
        )
        print(state_dict)
        model.load_state_dict(state_dict)
    return model
