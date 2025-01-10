# Experiments on LIVEC dataset

## train clean model
CUDA_VISIBLE_DEVICES=0 python train_test_IQA_sparsity.py


## train surrogate model using partial data
CUDA_VISIBLE_DEVICES=0 python train_test_IQA_sparsity_partial.py --partial_rate 0.2 --train_patch_num 5

## search trigger (Algorithm 1 UAP-DCT in the paper)
CUDA_VISIBLE_DEVICES=0 python UAP_DCT_demo.py --epsilon 8.0 --epoch 50 --partial_rate 0.2 --optimizer adam --model_path ./checkpoints/partial0.2_livec_bs32_grad[0]_weight[0.0].pth

## search adversarial perturbations for C-BAIQA (eps=2/255, typo mistakes in the paper)
CUDA_VISIBLE_DEVICES=0 python PGD_demo.py --eps 0.00784 --partial_rate 0.2 --model_path ./checkpoints/partial0.2_livec_bs32_grad[0]_weight[0.0].pth


## P-BAIQA (3 IQA models)
CUDA_VISIBLE_DEVICES=0 python train_test_IQA_sparsity_backdoor_multi_v6.py --poison --poison_rate 0.2 --score_p_scale 10 --multi_range
CUDA_VISIBLE_DEVICES=0 python train_test_IQA_sparsity_backdoor_multi_v6.py --model DBCNN --epochs 16 --lr 0.001 --poison --poison_rate 0.2 --score_p_scale 10 --multi_range
CUDA_VISIBLE_DEVICES=0 python train_test_IQA_sparsity_backdoor_multi_v6.py --epoch 12 --batch_size 32 --model TReS --poison --poison_rate 0.2 --score_p_scale 10 --multi_range


## C-BAIQA (3 IQA models)
CUDA_VISIBLE_DEVICES=0 python train_test_IQA_sparsity_backdoor_multi_cl_v3.py --mean_score 50.0 --epsilon 8 --poison --poison_rate 0.2 --score_p_scale 10 --adp_path ./pgd_target_delta_e2_p0.2.pt
CUDA_VISIBLE_DEVICES=0 python train_test_IQA_sparsity_backdoor_multi_cl_v3.py --mean_score 50.0 --epsilon 8 --model DBCNN --epochs 16 --lr 0.001 --poison --poison_rate 0.2 --score_p_scale 10 --adp_path ./pgd_target_delta_e2_p0.2.pt
CUDA_VISIBLE_DEVICES=0 python train_test_IQA_sparsity_backdoor_multi_cl_v3.py --mean_score 50.0 --epsilon 8 --model TReS --epochs 12 --poison --poison_rate 0.2 --score_p_scale 10 --adp_path ./pgd_target_delta_e2_p0.2.pt
