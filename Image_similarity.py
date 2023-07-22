import matplotlib.pyplot as plt
import cv2
import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from kornia_moons.feature import *
from viz import draw_LAF_matches
from flask import Flask, request,jsonify
import os

UPLOAD_FOLDER = './upload'
output_folder="./output"
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#!python -c "import kornia; print(kornia.__version__)"


def load_img(img):
  im1 = cv2.imread(img)
  im1re = cv2.resize(im1, (700, 700)).astype(np.double() )

  im1re=im1re.transpose(2,0,1)
  im1re=torch.tensor(im1re)

  im1re= torch.unsqueeze(im1re,0)
  im1re=im1re.type(torch.DoubleTensor)

  return im1re


@app.route('/getmatches', methods=['GET', 'POST'])
def inference():

    img1 = request.files.get('image1')
    img2 = request.files.get('image2')

    fname1 = UPLOAD_FOLDER+'/'+img1.filename
    fname2 = UPLOAD_FOLDER+'/'+img2.filename

    img1.save(fname1)
    img2.save(fname2)

    img1 = load_img(fname1)
    img2 = load_img(fname2)

    matcher = KF.LoFTR(pretrained='outdoor')

    input_dict = {"image0": K.color.rgb_to_grayscale(img1.type(torch.DoubleTensor)), # LofTR works on grayscale images only
                "image1": K.color.rgb_to_grayscale(img2.type(torch.DoubleTensor))}

    print("input_dict :",input_dict['image0'].shape)

    with torch.no_grad():
        correspondences = matcher(input_dict)
    mkpts0 = correspondences['keypoints0'].cpu().numpy()
    mkpts1 = correspondences['keypoints1'].cpu().numpy()
    H, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.5, 0.999, 100000)
    inliers = inliers > 0
    fig, ax = plt.subplots()
    # Perform matching using LoFTR
    draw_LAF_matches(
        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts0).view(1,-1, 2),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts0.shape[0]).view(1,-1, 1)),

        KF.laf_from_center_scale_ori(torch.from_numpy(mkpts1).view(1,-1, 2),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1, 1),
                                    torch.ones(mkpts1.shape[0]).view(1,-1, 1)),
        torch.arange(mkpts0.shape[0]).view(-1,1).repeat(1,2),
        K.tensor_to_image(img1),
        K.tensor_to_image(img2),
        inliers,
        draw_dict={'inlier_color': (0.2, 1, 0.2),
                'tentative_color': None,
                'feature_color': (0.2, 0.5, 1), 'vertical': False}, ax=ax)
    output_fname1=output_folder+img1.filename
    fig.savefig(output_fname1,dpi=110,bbox_inches='tight')

      # Remove the temporary uploaded images
    os.remove(fname1)
    os.remove(fname2)


    return jsonify({"output_image_path": output_fname1})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)