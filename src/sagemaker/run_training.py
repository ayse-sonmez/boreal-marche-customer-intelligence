from sagemaker.core import image_uris
from sagemaker.core.helper.session_helper import Session
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import (
    Compute,
    InputData,
    SourceCode,
    StoppingCondition,
)


REGION = "ca-central-1"

ROLE_ARN = (
    "arn:aws:iam::213104855806:"
    "role/BorealMarche-SageMaker-ExecutionRole"
)

BUCKET = "boreal-marche-ayse-213104855806"

TRAIN_DATA = (
    f"s3://{BUCKET}/data/train/"
    "customer_features_train.parquet"
)


def main():
    session = Session()

    training_image = image_uris.retrieve(
        framework="sklearn",
        region=REGION,
        version="1.4-2",
        py_version="py3",
        instance_type="ml.m5.large",
        image_scope="training",
    )

    print(f"Training image: {training_image}")

    source_code = SourceCode(
        source_dir="src/sagemaker",
        entry_script="train.py",
    )

    compute = Compute(
        instance_type="ml.m5.large",
        instance_count=1,
        enable_managed_spot_training=True,
    )

    trainer = ModelTrainer(
        training_image=training_image,
        source_code=source_code,
        compute=compute,
        stopping_condition=StoppingCondition(
            max_runtime_in_seconds=600,
            max_wait_time_in_seconds=900,
        ),
        role=ROLE_ARN,
        base_job_name="boreal-marche-retention",
        sagemaker_session=session,
        hyperparameters={
            "n-estimators": 157,
            "learning-rate": 0.01436244069943827,
            "max-depth": 3,
        },
    )

    train_data = InputData(
        channel_name="train",
        data_source=TRAIN_DATA,
    )

    print("Starting SageMaker Managed Spot training...")

    trainer.train(
        input_data_config=[train_data],
        wait=True,
        logs=True,
    )

    print("SageMaker training completed.")

    model_uri = (
        trainer._latest_training_job
        .model_artifacts
        .s3_model_artifacts
    )

    print(f"Model artifact: {model_uri}")


if __name__ == "__main__":
    main()