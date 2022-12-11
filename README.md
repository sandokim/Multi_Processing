간략하게

rank는 개별 gpu 넘버 / world size는 전체 gpu 개수

multi-gpus로 학습하는 경우, 배치사이즈별로 나눠서 여러개 gpu로 학습할 수 있음

ex) 2개의 서버 / 각 서버는 4개의 gpu를 가진다 가정하면
총 8개의 gpu가 있고, world size는 따라서 8개

ex) batch size  = 32이고 world size가 8이면 
배치사이즈를 world size로 나눈 만큼 gpu 8개가 나눠서 학습
32/8=4 이므로 batch size=4인 (데이터 4개씩) 배치를 gpu 8개가 동시에 학습

rank는 8이고 rank는 [0,1,2,3,4,5,6,7]이 된다.
