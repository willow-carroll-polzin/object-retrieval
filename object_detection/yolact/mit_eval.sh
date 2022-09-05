python3 -W ignore ./eval.py \
	--trained_model=./weights/yolact_darknet53_54_800000.pth \
	--top_k=5 \
	--cuda=True \
	--config=yolact_darknet53_config \
	--images=./data/archive/small_eval/:./data/images_out \
	--score_threshold=0.15
