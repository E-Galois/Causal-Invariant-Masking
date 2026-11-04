# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified by the Causal-Invariant-Masking contributors, 2026:
# renamed the builder to AdVQAGeneratorBasedBuilder, switched to repository-local
# data/AdVQA assets, retained validation only, and removed unused imports/comments.
# This file, including these modifications, is distributed under Apache-2.0.
# See licenses/Apache-2.0.txt and THIRD_PARTY_NOTICES.md at the repository root.
"""adVQA loading script."""
import json
from pathlib import Path
import datasets

_CITATION = "@InProceedings{sheng2021human,\nauthor = {Sheng, Sasha and Singh, Amanpreet and Goswami, Vedanuj and Magana, Jose Alberto Lopez and Galuba, Wojciech and Parikh, Devi and Kiela, Douwe},\ntitle = {Human-Adversarial Visual Question Answering},\njournal={arXiv preprint arXiv:2106.02280},\nyear = {2021},\n} \n"
_DESCRIPTION = "This is v1.0 of the ADVQA dataset.\n"
_HOMEPAGE = "https://adversarialvqa.org"
_LICENSE = "CC BY-NC 4.0"  # Dataset metadata, not the license of this loader code.
_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "AdVQA"
_URLS = {
    "questions": {
        "val": str(_DATA_ROOT / "v1_OpenEnded_mscoco_val2017_advqa_questions.json")
    },
    "annotations": {
        "val": str(_DATA_ROOT / "v1_mscoco_val2017_advqa_annotations.json")
    },
    "images": {"val": str(_DATA_ROOT / "val2014.zip")},
}
_SUB_FOLDER_OR_FILE_NAME = {
    "questions": {"val": None},
    "annotations": {"val": None},
    "images": {"val": "val2014"},
}


class AdVQAGeneratorBasedBuilder(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features(
            {
                "answers": [
                    {
                        "answer": datasets.Value("string"),
                        "answer_id": datasets.Value("int64"),
                    }
                ],
                "image_id": datasets.Value("int64"),
                "answer_type": datasets.Value("string"),
                "question_id": datasets.Value("int64"),
                "question": datasets.Value("string"),
                "image": datasets.Image(),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URLS)
        gen_kwargs = {}
        for split_name in ["val"]:
            gen_kwargs_per_split = {}
            for dir_name in _URLS.keys():
                sub_folder_or_file_name = _SUB_FOLDER_OR_FILE_NAME.get(
                    dir_name, None
                ).get(split_name, None)
                if (
                    split_name in data_dir[dir_name]
                    and sub_folder_or_file_name is not None
                ):
                    path = (
                        Path(data_dir[dir_name][split_name]) / sub_folder_or_file_name
                    )
                elif split_name in data_dir[dir_name]:
                    path = Path(data_dir[dir_name][split_name])
                else:
                    path = None
                gen_kwargs_per_split[f"{dir_name}_path"] = path
            gen_kwargs[split_name] = gen_kwargs_per_split
        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs=gen_kwargs["val"]
            )
        ]

    def _generate_examples(self, questions_path, annotations_path, images_path):
        questions = json.load(open(questions_path, "r"))
        dataset = json.load(open(annotations_path, "r"))
        qa = {ann["question_id"]: [] for ann in dataset["annotations"]}
        for ann in dataset["annotations"]:
            qa[ann["question_id"]] = ann
        for question in questions["questions"]:
            annotation = qa[question["question_id"]]
            assert (
                len(set(question.keys()) ^ set(["image_id", "question", "question_id"]))
                == 0
            )
            assert (
                len(
                    set(annotation.keys())
                    ^ set(["answers", "image_id", "answer_type", "question_id"])
                )
                == 0
            )
            record = question
            record.update(annotation)
            record["image"] = str(
                images_path / f"COCO_{images_path.name}_{record['image_id']:0>12}.jpg"
            )
            yield (question["question_id"], record)
