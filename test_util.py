import random

import pytest

import util


def test_update_ratings():
    model_ratings = [
        ("model1", 1000),
        ("model2", 1000),
        ("model3", 1000),
        ("model4", 1000),
    ]
    updated = util.update_ratings(0, model_ratings)
    assert len(updated) == 4
    assert updated[0][0] == "model1"
    assert (
        updated[0][1] > 1000
    ), f"Expected winner's rating to increase, but got {updated[0][1]}"
    assert all(
        rating < 1000 for _, rating in updated[1:]
    ), "Expected all other ratings to decrease"

    # Test case 2: 10 interations
    model_ratings = [
        ("model1", 1000),
        ("model2", 1000),
        ("model3", 1000),
        ("model4", 1000),
    ]
    for _ in range(10):
        model_ratings = util.update_ratings(0, model_ratings)

    assert model_ratings[0][0] == "model1"
    assert (
        model_ratings[0][1] > 1200
    ), "Expected winner's rating to increase significantly after 10 wins"
    assert all(
        rating < 1000 for _, rating in model_ratings[1:]
    ), "Expected losing models' ratings to decrease after 10 losses"

    # Test case 2: 10 interations
    model_ratings = [
        ("model1", 1000),
        ("model2", 1000),
        ("model3", 1000),
        ("model4", 1000),
        ("model5", 1000),
        ("model6", 1000),
        ("model7", 1000),
        ("model8", 1000),
    ]
    for _ in range(100):
        models_to_rate = random.sample(model_ratings, 4)
        updated_ratings = util.update_ratings(random.randint(0, 3), models_to_rate)

        # Update the ratings in the original model_ratings list
        for updated_model, updated_rating in updated_ratings:
            for i, (model, _) in enumerate(model_ratings):
                if model == updated_model:
                    model_ratings[i] = (model, updated_rating)
                    break

    print(model_ratings)


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__])
