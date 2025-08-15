include .env


.PHONY: create-notebook extract clean


NOTEBOOK_INSTANCE_NAME=llm-data-extraction-notebook

CONFIG_NAME=clone-llm-repo-config
CONFIG_NAME_ARG=--notebook-instance-lifecycle-config-name "$(CONFIG_NAME)"


INSTANCE_SPECS=--instance-type "$(INSTANCE_TYPE)" --volume-size-in-gb $(VOLUME_SIZE)
INSTANCE_NAMES=--notebook-instance-name "$(NOTEBOOK_INSTANCE_NAME)" --lifecycle-config-name "$(CONFIG_NAME)"


create-notebook:
	aws sagemaker create-notebook-instance --role-arn "$(ROLE_ARN)" $(INSTANCE_NAMES) $(INSTANCE_SPECS)

extract: create-notebook
	aws sagemaker create-notebook-instance-lifecycle-config $(CONFIG_NAME) --on-create Content=$(ON_CREATE)

clean:
	aws sagemaker stop-notebook-instance --notebook-instance-name "$(NOTEBOOK_INSTANCE_NAME)"
	aws sagemaker delete-notebook-instance --notebook-instance-name "$(NOTEBOOK_INSTANCE_NAME)"
	aws sagemaker delete-notebook-instance-lifecycle-config $(CONFIG_NAME_ARG)

