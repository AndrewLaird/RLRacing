from CarRacingModel import CarRacingModel
from Environment import CarRacingStacked


if __name__ == "__main__":
    num_stacked=5
    env = CarRacingStacked(num_stacked=num_stacked)
    model = CarRacingModel(num_layers=num_stacked)