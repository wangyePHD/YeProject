import fiftyone


dataset = fiftyone.zoo.load_zoo_dataset(
              "open-images-v6",
              split="test",
              classes=["Car"]
          )
