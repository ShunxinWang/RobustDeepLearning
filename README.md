# RobustDeepLearning
## Models can be trained by using command line:
`python train.py --model_depth AE_L3 --image_size 32 --num_class 10 --num_classifiers 1 --shortcut True --weight_mode None --weight_alpha 1 --weight_beta 1`

**`--model_depth: AE/EN_L1/2/3/4`
**`--image_size: 32/64/224`
**`--num_class: 10/200/200`
**`--num_classifiers: 1/4`
**`--shortcut: True/False`
**`--weight_mode: None/fixed_schedule_(alpha/beta)/dynamic_(alpha/beta)`
**`--weight_alpha: 1`
**`--weight_beta: 1`
