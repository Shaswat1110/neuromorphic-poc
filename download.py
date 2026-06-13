import tonic

# Override default URLs to bypass the AWS WAF block
tonic.datasets.DVSGesture.train_url = "https://ndownloader.figshare.com/files/38022171"
tonic.datasets.DVSGesture.test_url = "https://ndownloader.figshare.com/files/38020584"

print("Initiating dataset download via direct subdomain...")
tonic.datasets.DVSGesture(save_to='./datasets')
print("Download and extraction complete.")