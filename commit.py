import json
import base64
import requests


def push_to_github(filename, branch, token):
    url="https://api.github.com/repos/scy6500/token_test/contents/" + filename
    response = requests.get(url,
                            headers={'Accept': 'application/vnd.github.v3+json'}).json()
    sha = response["sha"]
    base64content=base64.b64encode(open(filename,"rb").read())
    message = json.dumps({"message": "update",
                          "branch": branch,
                          "content": base64content.decode("utf-8"),
                          "sha": sha
                          })
    resp = requests.put(url, data=message,
                        headers={"Content-Type": "application/json", "Authorization": "token " + token})
    print(resp.json())


def main():
    token = ""
    files= ["data/minmax.json", "data/std.json", "model/model.pt", "model_info.json"]
    branch= "main"
    for file in files:
        push_to_github(file, branch, token)
