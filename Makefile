# Makefile for VLM Grasp project
# Quick shortcuts for common tasks

.PHONY: setup check test-loader test-overfit train eval clean help

help:
	@echo "VLM Grasp Project - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup         - Install dependencies"
	@echo "  check         - Run setup verification"
	@echo "  test-loader   - Test data loading"
	@echo "  test-overfit  - Run overfitting test (sanity check)"
	@echo "  train         - Start full training"
	@echo "  eval          - Run evaluation"
	@echo "  clean         - Clean generated files"
	@echo ""

setup:
	pip install -r requirements.txt

check:
	python setup_check.py

test-loader:
	python data/test_loader.py

test-overfit:
	python model/test_overfit.py --n_samples 10 --n_steps 200

train:
	python train/train_grasp_lora.py \
		--data_dir ./ocid-grasp \
		--checkpoint_dir ./checkpoints/grasp-vlm-$(shell date +%Y%m%d_%H%M%S)

eval:
	python eval/evaluate_all.py \
		--checkpoint ./checkpoints/grasp-vlm-lora/final \
		--baselines

clean:
	rm -rf checkpoints/*
	rm -rf results/*
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -f data_loader_test.png
