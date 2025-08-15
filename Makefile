include .env


.PHONY: create-notebook extract clean git-config create-sagemaker-role delete-sagemaker-role


NOTEBOOK_INSTANCE_NAME=llm-data-extraction-notebook

CONFIG_NAME=clone-llm-repo-config
CONFIG_NAME_ARG=--notebook-instance-lifecycle-config-name "$(CONFIG_NAME)"


INSTANCE_SPECS=--instance-type "$(INSTANCE_TYPE)" --volume-size-in-gb $(VOLUME_SIZE)
INSTANCE_NAMES=--notebook-instance-name "$(NOTEBOOK_INSTANCE_NAME)" --lifecycle-config-name "$(CONFIG_NAME)"


SAGEMAKER_ROLE_NAME := SageMaker-Execution-Role
SAGEMAKER_POLICY_ARN := arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

create-sagemaker-role:
	@echo "Creating IAM role: $(SAGEMAKER_ROLE_NAME)..."
	@aws iam create-role --role-name $(SAGEMAKER_ROLE_NAME) --assume-role-policy-document file://extraction/sagemaker-iam.json
	
	@echo "Attaching policy: AmazonSageMakerFullAccess..."
	@aws iam attach-role-policy --role-name $(SAGEMAKER_ROLE_NAME) --policy-arn $(SAGEMAKER_POLICY_ARN)
	
	@echo "Role created successfully. Use this name in your other commands."

delete-sagemaker-role:
	@echo "Detaching policy from $(SAGEMAKER_ROLE_NAME)..."
	@aws iam detach-role-policy --role-name $(SAGEMAKER_ROLE_NAME) --policy-arn $(SAGEMAKER_POLICY_ARN)
	    
	@echo "Deleting IAM role: $(SAGEMAKER_ROLE_NAME)..."
	@aws iam delete-role --role-name $(SAGEMAKER_ROLE_NAME)
	
	@echo "Role deleted successfully."


create-notebook:
	aws sagemaker create-notebook-instance --role-arn "$(ROLE_ARN)" $(INSTANCE_NAMES) $(INSTANCE_SPECS)

extract: create-notebook
	aws sagemaker create-notebook-instance-lifecycle-config $(CONFIG_NAME) --on-create Content=$(ON_CREATE)

clean:
	aws sagemaker stop-notebook-instance --notebook-instance-name "$(NOTEBOOK_INSTANCE_NAME)"
	aws sagemaker delete-notebook-instance --notebook-instance-name "$(NOTEBOOK_INSTANCE_NAME)"
	aws sagemaker delete-notebook-instance-lifecycle-config $(CONFIG_NAME_ARG)
	aws iam delete-role --role-name $(SAGEMAKER_ROLE_NAME)


git-config:
	git config --global user.name "$(USER_NAME)"
	git config --global user.email "$(USER_EMAIL)"
	git add $(NOTEBOOK_NAME).ipynb
	git commit -m "Add initial data exploration notebook."
	git push
