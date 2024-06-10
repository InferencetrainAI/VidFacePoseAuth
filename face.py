#Disclaimer this has not been tested well

import os
import cv2
from insightface.app import FaceAnalysis
import torch

import sys
args = sys.argv


# prompt: compare face embediggs




class FaceRec:
    def __init__(self):
        self.foldername = '/home/emmanuel-nsanga/Pictures/Webcam'
        self.files = []
        self.files_attempt = []
        self.embeds = []
        self.diff = []
        self.ground_mathches = []
        self.sample_true = []
        self.sample_attemt = []
        self.folder_attempt='/home/emmanuel-nsanga/Pictures/Webcam/'
        self.folder_ground = '/home/emmanuel-nsanga/Pictures/webcam/'
        self.folder_camera = '/home/emmanuel-nsanga/Pictures/camera/'
        self.files_ground = [self.folder_ground+files for files in os.listdir(self.folder_ground)]
        self.files_attempt = [self.folder_attempt+files for files in os.listdir(self.folder_attempt)]
        self.files_camera = [self.folder_camera+files for files in os.listdir(self.folder_camera)]
        self.zip_ground = list(zip(self.files_ground, self.files_attempt))
        self.zip_attempt = list(zip(self.files_attempt, self.files_camera))


        
    
    def embeddings(self, image):
        app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        image1 = cv2.imread(image)
        faces = app.get(image1)

        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
        return(torch.Tensor(faceid_embeds))



    def face_embed(self, face, face1):
        # Load the two images and get their face embeddings.
        face_encodings = self.embeddings(face)
        face_encodings1 = self.embeddings(face1)
        return(torch.nn.functional.cosine_similarity(face_encodings, face_encodings1))
    


    
    

    def expectation(self, sample_data):
        mean, std = torch.mean(sample_data), torch.std(sample_data)
        distribute = torch.distributions.Normal(mean, std)
        return(distribute.sample(sample_shape=(10,)))




    def sim_distribution(self):
        ground_embeddings = self.zip_attempt[len(self.zip_ground)::]
        
        

        w_r_t_g = self.zip_ground[0::]
        w_r_t_c = self.zip_attempt

        w_r_t_g = self.zip_ground[0::len(self.zip_ground)//2]
        w_r_t_tr = self.zip_ground[len(self.zip_ground)//2::]


        

        ground_embeddings = [self.face_embed(attempting, attempt) for attempting, attempt in w_r_t_g]
        attempt_ground = [self.face_embed(attempting, attempt) for attempting, attempt in w_r_t_tr]


        ground_embeddings_g = [self.face_embed(attempting, attempt) for attempting, attempt in w_r_t_g]
        attempt_ground_c = [self.face_embed(attempting, attempt) for attempting, attempt in w_r_t_c]


                
        self.sampling_ground = self.expectation(torch.Tensor(ground_embeddings))
        self.sampling_attempt_g = self.expectation(torch.Tensor(attempt_ground))


        self.sampling_ground = self.expectation(torch.Tensor(ground_embeddings_g))
        self.sampling_attempt_c = self.expectation(torch.Tensor(attempt_ground_c))

        return(self.sampling_ground, self.sampling_attempt_g, self.sampling_ground, self.sampling_attempt_c)
    

    
    def model(self, max_itter=3):       

        booleans = []

        itter = 0
        while itter < max_itter:
            if itter <= max_itter:
                sim_distribution = self.sim_distribution()
                xy = torch.mean(torch.Tensor([x-y for x, y in zip(sim_distribution[2], sim_distribution[3])]))
                print(xy.item())

                if xy.item() < 0.4:
                    booleans.append(1)      


                else:
                    pass

            itter+=1


        if sum(booleans) > 0:
            os.system('echo 'pass' | -S sudo command") #after su access in by low root privileges echo 'real password' for elevated root access //
                                    #login with very low root access - disclaimer 
            
            
        else:
            print('False')


        

    def camera(self):
        

Recognition = FaceRec()
print(Recognition.model())

