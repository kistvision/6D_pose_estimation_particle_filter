# 6D_pose_estimation_particle_filter

## Requirements
 * __CMake >= 3.17__
 * __Python >= 3.6__
 * __CUDA >= 10.2__     (We tested on CUDA 10.2)
 * __Pytorch >= 1.5.0__ (We tested on Pytorch 1.5 and 1.7.1 and 1.9)

## Installation

Change compile.sh line 5 to the glm library include path. This library can be downloaded from this [link](https://github.com/g-truc/glm).

    $ cd lib
    $ sh compile.sh
    $ cd ..
    $ conda env create -f requirements.yaml
    $ conda activate pf_with_cpn

Or you can just run below in your own anaconda environment.

    $ pip install opencv-python transforms3d open3d scipy

## How to make predefined grasp poses
### Convert obj file to pcd file

    $ cd lib/obj2pcd && mkdir build && cd build && cmake .. && make
    $ ./obj2pcd <obj file path>

### Predefine grasp pose
Install [graspnetAPI](https://github.com/graspnet/graspnetAPI)
Write your code (grasp pose x,y,z,roll,pitch,yaw ...) in function `vis_predefined_grasps`

    $ python grasp_pose_predefined.py <pcd file path>



## 하이퍼파라미터
dataset, save_path, w_o_CPN 등은 논문 작성을 위한 evaluation에 쓰였던 파라미터이고, 실제 활용할 때에는 아래 파라미터만 참고하면 됨
 * tau(float) : 클수록 likelihood 계산시에 마스크 영역에 더 정확하게 맞추려는 경향이 있고, 작을수록 거리를 더 정확히 맞추려는 경향이 있음. 자세한 내용은 논문을 참고해주세요.
 * num_particles(int) : 파티클 필터에서 활용할 파티클 개수
 * visualization(True/False) : 결과화면을 띄울 것인지 여부

## 실행 방법
6D object pose tracking 예제:
(Realsense SDK 설치하고, Realsense 카메라 연결 후 실행)

    $ ./tracking.sh

RGB 이미지, Depth 이미지, Mask 이미지 파일을 가지고 6D pose estimation 하는 예제:

    $ ./run.sh

## 그 외
이론적인 내용은 [논문](https://ieeexplore.ieee.org/document/10034745/)을 참고하시고, 코드를 활용하기 위해서는 input (RGB, depth, mask 이미지) 형식 잘 맞추고, particle_filter.py의 클래스 초기화 하는 부분에서 CAD 모델 등록하는 부분만 참고해서 바꿔주면 됩니다.

그리퍼 포즈 pre-define을 하기 위해서는 grasp_pose_predefine.py를 실행하고 눈으로 보면서 코드에 그리퍼 포즈를 적절히 수정하며 맞춰주면 됩니다. 이 방법은 정상적인 방법은 아니며, 제가 편해서 이렇게 코드를 작성한 것이니, 사용자가 편한 방법대로 그리퍼의 포즈를 사전 정의하면 됩니다. 

YOLACT 가중치파일 용량이 커서 따로 다운받아 `yolact/weights` 경로에 넣어주세요. [여기](https://drive.google.com/file/d/1spkxoWMWks7gUC_O01JvReRq8ZMIisb1/view?usp=sharing)에서 다운받거나 NAS 서버 `[인수인계]/gijae/6D object pose estimation` 경로에서 yolact_resnet50_204_90000.pth 파일을 받으세요.
