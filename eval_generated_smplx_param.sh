# python eval_generated_smplx_param.py \
#     --checkpoint /scratch/rhong5/weights/temp_training_weights/aslAvatar/ASLAvatar_Diffusion/ASL3DWord/20260226_223940_job6035473/best_model.pt \
#     --dataset_name ASL3DWord \
#     --use_upper_body \
#     --eval_only \
#     --use_rot6d \
#     --use_phono_attribute


# python eval_generated_smplx_param.py \
#     --checkpoint /scratch/rhong5/weights/temp_training_weights/aslAvatar/ASLAvatar_Diffusion/ASL3DWord/20260302_152358_job6201204/best_model.pt \
#     --dataset_name ASL3DWord \
#     --use_upper_body \
#     --eval_only \
#     --use_rot6d \
#     --use_phono_attribute \
#     --text_encoder_type t5



python eval_generated_smplx_param.py \
    --checkpoint /scratch/rhong5/weights/temp_training_weights/aslAvatar/ASLAvatar_Diffusion/ASL3DWord/20260310_030439_job6539103/best_model.pt \
    --dataset_name ASL3DWord \
    --use_upper_body \
    --eval_only \
    --use_rot6d \
    --text_encoder_type t5

