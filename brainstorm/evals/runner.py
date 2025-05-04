import asyncio
import json

from sqlalchemy import select
from starlette.responses import StreamingResponse

from brainstorm.core.adapters import get_model_adapter
from brainstorm.datasets.initial_datasets import DATASETS_MAP
from brainstorm.db.models.evals import EvalDefinition
from brainstorm.db.models.model import ModelDefinition
from brainstorm.scorers.scorers import SCORERS_MAP


async def run_evals(eval_id: str, db):
    """
    get eval def
    from there get dataset def
    from there get model def
    from there get scorer def

    have some map of dataset adapters to dataset objects
    instantiate object

    instantiate model adapter from model def


    from eval def get scorer (also via map)

    for each row in the dataset
        call model adapter with the input field
        call scorer score row

    call scorer agg
    save and return eval result
    """

    eval_stmt = select(EvalDefinition).filter_by(eval_id=eval_id)
    eval_definition = db.execute(eval_stmt).scalar_one_or_none()

    # todo: we should do this via defaults or something, don't like messing with def after its retrieved
    if eval_definition.scorer_agg_dimensions is None or eval_definition.scorer_agg_dimensions == '':
        eval_definition.scorer_agg_dimensions = ['dataset_split_key']
    elif 'dataset_split_key' not in eval_definition.scorer_agg_dimensions:
        eval_definition.scorer_agg_dimensions.append('dataset_split_key')
    # todo: also prevent this at model and UI levels
    eval_definition.scorer_agg_dimensions = [i for i in eval_definition.scorer_agg_dimensions if i != '']

    # model_stmt = select(ModelDefinition).filter_by(model_id=eval_definition.model_id)
    model_definition = eval_definition.model_definition
    dataset_definition = eval_definition.dataset_definition

    scorer = SCORERS_MAP[eval_definition.scorer](
        agg_functions=eval_definition.scorer_agg_functions,
        agg_dimensions=eval_definition.scorer_agg_dimensions
    )
    dataset = DATASETS_MAP[dataset_definition.dataset_adapter](dataset_definition=dataset_definition)
    model = get_model_adapter(model_definition)

    predictions = []

    async def get_and_score_prediction(dataset_row):
        prediction = await model.generate(dataset_row['input'])
        predictions.append(prediction)
        result_score = await scorer.score_row(dataset_row=dataset_row, prediction=prediction)
        return prediction, result_score

    # promises = []
    # for split_key in dataset.keys():
    #     dataset_split = dataset[split_key]
    #     for row in dataset_split:
    #         row['dataset_split_key'] = split_key
    #         promises.append(get_and_score_prediction(dataset_row=row))
    #
    # await asyncio.gather(*promises)
    #
    # scorer.aggregate_scores()

    eval_definition.results = scorer.agg_scores

    # todo: there should also be detailed score/result store and endpoint which would show exact score, response etc
    #  for each dataset row for inspection in the UI

    all_dicts = []

    async def result_generator():
        for split_key in dataset.keys():
            dataset_split = dataset[split_key]
            dataset_split = dataset_split.map(lambda x: {'dataset_split_key': split_key})

            for row in dataset_split:
                prediction, score = await get_and_score_prediction(dataset_row=row)
                response = {
                    'input': row['input'],
                    'output': row['output'],
                    'prediction': prediction,
                    'score': score,
                }
                all_dicts.append(response)
                json_string = json.dumps(response) + "\n"

                yield json_string
        else:
            scorer.aggregate_scores()
            all_dicts.append(scorer.agg_scores)
            json_string = json.dumps(scorer.agg_scores) + "\n"

            eval_definition.results = all_dicts
            db.merge(eval_definition)
            db.commit()

            yield json_string

    return result_generator

