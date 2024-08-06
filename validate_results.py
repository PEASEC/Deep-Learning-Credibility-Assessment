import sys
import os
from src.ml.execute import Executable, FeatureTypes, advanced_timeline_feature_wrapper
from typing import Union, ClassVar, Dict
from src.twitter import TweetDataset
from src.util import get_repository_path


def get_optional_argument_by_name(arg_name: str) -> Union[str, bool, None]:
    arg_name = "--" + arg_name
    for arg in sys.argv:
        if arg.startswith(arg_name):
            value = arg[len(arg_name):]
            if len(value) <= 0:
                return True

            if value.startswith(":") or value.startswith("="):
                main_value = value[1:]
                if main_value.lower() in ["on", "true", "yes"]:
                    return True
                elif main_value.lower() in ["off", "false", "no"]:
                    return False
                else:
                    return main_value

    return None


def get_argument_by_index(index: int) -> Union[str, None]:
    if index >= len(sys.argv):
        return None

    for arg in sys.argv:
        if not arg.startswith("-"):
            index -= 1
            if index < 0:
                return arg

    return None


def get_desired_type_from_name(name: str) -> FeatureTypes:
    if name is None:
        return FeatureTypes.BASIS_ONLY

    name = name.replace(".", "-")
    name = name.replace("_", "-")
    name = name.replace(":", "-")
    name = name.replace(",", "-")
    name_list = [n.lower() for n in name.split("-")]
    name_list.sort()
    return FeatureTypes("-".join(name_list))


def get_desired_type() -> FeatureTypes:
    if len(sys.argv) <= 3:
        return FeatureTypes.BASIS_ONLY
    else:
        try:
            return get_desired_type_from_name(get_argument_by_index(3))
        except ValueError:
            print("Unknown feature type '{}'.".format(get_argument_by_index(3)))
            exit()


def get_database_path() -> str:
    path = get_optional_argument_by_name("dataset")
    if path is None:
        return get_repository_path("datasets/credibility-large")
    else:
        if path == "large":
            path = "credibility-large"
        elif path == "default":
            path = "credibility-default"

        local_path = get_repository_path(os.path.join("datasets", path), return_none_on_error=True)
        if local_path is not None:
            return local_path
        else:
            return get_repository_path(path)


def get_path_name(database_path: str) -> str:
    database_path = os.path.basename(database_path)
    database_path = database_path.replace("\\", "-")
    database_path = database_path.replace("/", "-")
    database_path = database_path.replace(":", "")
    return database_path


def get_executable(
        dataset: TweetDataset,
        model_type: str,
        feature_type: FeatureTypes,
        variant: str,
        advanced_timeline: Union[bool, str, None],
        display_progress: bool = True
) -> Executable:
    def modify_if_advanced_timeline_mode(executor: ClassVar[Executable]) -> ClassVar[Executable]:
        if advanced_timeline:
            if isinstance(advanced_timeline, str):
                return advanced_timeline_feature_wrapper(
                    executor,
                    advanced_timeline,
                    get_executable(dataset, advanced_timeline, FeatureTypes.TEXT_TWEET_USER, variant, False, False)
                )
            else:
                return advanced_timeline_feature_wrapper(executor)
        else:
            return executor

    if model_type == "bert":
        from src.ml.execute import BertExecutable
        executable_class = modify_if_advanced_timeline_mode(BertExecutable)

        bert_path = get_optional_argument_by_name("bert-model")
        if bert_path is None:
            bert_path = "./resources/bert/" + dataset_name
        bert_path = get_repository_path(bert_path)

        return executable_class(
            model_path=working_directory,
            pretained_bert_path=bert_path,
            feature_type=feature_type,
            display_progress=display_progress,
            variant=dataset_name
        )
    elif model_type == "rnn":
        from src.ml.execute import RnnExecutable
        executable_class = modify_if_advanced_timeline_mode(RnnExecutable)

        keep_embeddings_in_memory = get_optional_argument_by_name("memory")
        embeddings_path = get_repository_path(get_optional_argument_by_name("embeddings"))

        if keep_embeddings_in_memory is None:
            keep_embeddings_in_memory = False

        return executable_class(
            model_path=working_directory,
            feature_type=feature_type,
            display_progress=display_progress,
            variant=dataset_name,
            keep_embeddings_in_memory=keep_embeddings_in_memory,
            embedding_file=embeddings_path
        )
    elif model_type == "mlp":
        from src.ml.execute import MlpExecutable
        executable_class = modify_if_advanced_timeline_mode(MlpExecutable)

        return executable_class(
            model_path=working_directory,
            feature_type=feature_type,
            display_progress=display_progress,
            variant=dataset_name
        )
    else:
        print(
            "Unknown model type '{}'. Please refer to the documentation.".format(get_argument_by_index(1))
        )
        exit()
        raise ValueError()


def get_working_directory(database_path: str) -> str:
    working_path = get_optional_argument_by_name("working-dir")
    if working_path is None:
        working_path = get_optional_argument_by_name("model-path")

    if working_path is None:
        d_name = get_path_name(database_path)
        working_path = os.path.join("build", d_name)

    return get_repository_path(working_path)


def analyze_data():
    d_path = get_database_path()
    dataset = TweetDataset(
        d_path,
        load_user_timeline=False,
        load_user_profile=False
    )

    print("Analyzing...")
    sources: Dict[str, list] = {}
    for entry in dataset:
        if entry.source in sources:
            sources[entry.source].append(entry.label)
        else:
            sources[entry.source] = [entry.label]

    total_negative = 0
    total_positive = 0
    for (src, values) in sources.items():
        negative = len([label for label in values if label < 0.5])
        positive = len([label for label in values if label > 0.5])

        total_negative += negative
        total_positive += positive

        print("{:>15}: Positive: {:>4}, Negative: {:>4}".format(src, positive, negative))

    print("{:>15}-------------------------".format("--------"))
    print("{:>15}: Positive: {:>4}, Negative: {:>4}".format("Total", total_positive, total_negative))
    print()

    train, dev, test = dataset.split()
    train_users = set([o.post.user_id for o in train])
    dev_users = set([o.post.user_id for o in dev])
    test_users = set([o.post.user_id for o in test])

    total_users = {*train_users, *dev_users, *test_users}

    print("{:>25}: {:>4} ({:>2.4f}%)".format(
        "Total unique users",
        len(total_users),
        len(total_users) / len(dataset) * 100
    ))
    train_dev_users = dev_users.intersection(train_users)
    print("{:>25}: {:>4} ({:>2.4f}%)".format(
        "Known users in dev set",
        len(train_dev_users),
        len(train_dev_users) / len(dev_users) * 100
    ))
    test_know_users = test_users.intersection(train_users.union(dev_users))
    print("{:>25}: {:>4} ({:>2.4f}%)".format(
        "Known users in test set",
        len(test_know_users),
        len(test_know_users) / len(test_users) * 100
    ))
    print()

    # Checking for duplicates
    set_1 = set(train.tweet_id_list)
    set_2 = set(dev.tweet_id_list)
    set_3 = set(test.tweet_id_list)
    set_all = set(dataset.tweet_id_list)

    if len(set_1.intersection(set_2)) > 0 or len(set_2.intersection(set_3)) > 0 or len(set_1.intersection(set_3)) > 0:
        print("Splitting-Health: [BROKEN] Not all sets are disjoint")
    elif len(set_1) + len(set_2) + len(set_3) != len(set_all):
        print("Splitting-Health: [BROKEN] Length of splitting sets does not match")
    else:
        print("Splitting-Health: [OK]")

    exit()


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Not enough arguments. Please provide the type (train, validate) and the basis model you want to use.")
        exit()

    if sys.argv[1] == "analyze":
        analyze_data()

    if len(sys.argv) <= 2:
        print("Not enough arguments. Please provide the basis model you want to use.")
        exit()

    model_type = get_argument_by_index(2).lower()
    feature_type = get_desired_type()
    dataset_path = get_database_path()
    working_directory = get_working_directory(dataset_path)
    dataset = TweetDataset(
        dataset_path,
        load_user_timeline="timeline" in feature_type.value,
        user_timeline_max_items=40,
        load_user_profile=False  # Never used, as also provided via tweet itself
    )
    dataset_name = get_path_name(working_directory)

    sample_count = get_optional_argument_by_name("sample")
    if sample_count is not None and int(sample_count) > 0:
        import random

        random.seed(42)
        dataset = dataset[random.sample(range(len(dataset)), int(sample_count))]

    advanced_timeline = get_optional_argument_by_name("advanced-timeline")
    executable = get_executable(dataset, model_type, feature_type, variant=dataset_name,
                                advanced_timeline=advanced_timeline)

    device_name = get_optional_argument_by_name("device")
    if device_name is not None:
        executable.device_name = device_name

    if get_optional_argument_by_name("force-cuda") or get_optional_argument_by_name("force-gpu"):
        executable.device_name = "cuda"
    elif get_optional_argument_by_name("force-cpu"):
        executable.device_name = "cpu"

    if sys.argv[1] == "train":
        ignore_cache = get_optional_argument_by_name("ignore-cache")
        if ignore_cache is None:
            ignore_cache = False

        executable.train(dataset, ignore_cache).plot()
    elif sys.argv[1] in ["validate", "val"]:
        dev_set_only = get_optional_argument_by_name("dev-set")
        test_set_only = get_optional_argument_by_name("test-set")

        if dataset.is_splittable and not dev_set_only and not test_set_only:
            print(
                "The selected dataset can be split into multiple subsets. " +
                "The validation will be executed on the WHOLE dataset. " +
                "To select a specific dataset provide --dev-set or --test-set.\n",
                file=sys.stderr
            )

        if dev_set_only and test_set_only:
            print(
                "The arguments --dev-set and --test-set are exclusive. " +
                "Please use only one of the two options.\n",
                file=sys.stderr
            )
            exit()

        if dev_set_only:
            dataset = dataset.split()[1]

        if test_set_only:
            dataset = dataset.split()[2]

        result_path = get_optional_argument_by_name("result-path")
        executable.validate(dataset, predictions_save_file=result_path)
    elif sys.argv[1] in ["time", "time-check", "time-analyze", "measure"]:
        wait_for_input = get_optional_argument_by_name("wait")
        executable.predict_and_measure_time(dataset)
        if wait_for_input:
            dataset = None
            import os

            print("My pid: {}".format(os.getpid()))
            input("Press Enter to continue...")

    else:
        print("Unknown action type '{}'. Use 'train' or 'validate'".format(get_argument_by_index(1)))
