import mlflow.azureml

from config import config
from config import model_config
from azureml.core import Workspace
from azureml.core.webservice import AciWebservice, Webservice

def run_deployment() -> None:
    """Train the model to Azure."""

    # Create or load an existing Azure ML workspace. You can also load an existing workspace using
    # Workspace.get(name=model_config.WORKSPACE_NAME,
    #                                 subscription_id=model_config.SUBSCRIPTION_ID,
    #                                 resource_group=model_config.RESOURCE_GROUP)
    azure_workspace = Workspace.get(name=model_config.WORKSPACE_NAME,
                                    subscription_id=model_config.SUBSCRIPTION_ID,
                                    resource_group=model_config.RESOURCE_GROUP)
    # azure_workspace = Workspace.create(name=model_config.WORKSPACE_NAME,
    #                                subscription_id=model_config.SUBSCRIPTION_ID,
    #                                resource_group=model_config.RESOURCE_GROUP,
    #                                location=model_config.LOCATION,
    #                                create_resource_group=False,
    #                                exist_okay=True)

    # Build an Azure ML container image for deployment
    model_path = config.S3_MODEL_PATH
    azure_image, azure_model = mlflow.azureml.build_image(model_uri=model_path,
                                                      workspace=azure_workspace,
                                                      description="Loan classification model 1",
                                                      synchronous=True)
    # If your image build failed, you can access build logs at the following URI:
    print("Access the following URI for build logs: {}".format(azure_image.image_build_log_uri))

    # Deploy the container image to ACI
    webservice_deployment_config = AciWebservice.deploy_configuration()
    webservice = Webservice.deploy_from_image(
                        image=azure_image, workspace=azure_workspace, name=model_config.DEPLOYMENT_NAME)
    webservice.wait_for_deployment()

    # After the image deployment completes, requests can be posted via HTTP to the new ACI
    # webservice's scoring URI. 
    print("Scoring URI is: %s", webservice.scoring_uri)

if __name__ == "__main__":
    run_deployment()
