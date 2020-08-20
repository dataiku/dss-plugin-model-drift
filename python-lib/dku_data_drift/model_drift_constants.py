# -*- coding: utf-8 -*-

class ModelDriftConstants(object):

    TIMESTAMP = 'timestamp'
    MODEL_ID = 'model_id'
    VERSION_ID = 'version_id'
    TRAIN_DATE = 'train_date'
    DRIFT_SCORE = 'drift_score'
    DRIFT_SCORE_DEFINITION = 'The drift score (between 0 and 1) is low (udner 0.1) if the new dataset and the original dataset are indistinguishable.'
    FUGACITY = 'fugacity'
    FUGACITY_CLASSIF_DEFINITION = 'Proportion of samples predicted (in %) in each class when scoring on both the original test and the new input dataset.'
    FUGACITY_REGRESSION_DEFINITION = 'Proportion of samples predicted (in %) in each decile when scoring on both the original test and the new input dataset.\n\n'
    FUGACITY_RELATIVE_CHANGE = 'fugacity_relative_change'
    FUGACITY_RELATIVE_CHANGE_CLASSIF_DEFINITION = 'Relative change (in %) in each class with respect to the original fugacity value.\n\nFormula: 100*(new_fugacity - original_fugacity)/original_fugacity'
    FUGACITY_RELATIVE_CHANGE_REGRESSION_DEFINITION = 'Relative change (in %) in each decile with respect to the original fugacity value.\n\nFormula: 100*(new_fugacity - original_fugacity)/original_fugacity\n\n'
    RISKIEST_FEATURES = 'riskiest_features'
    RISKIEST_FEATURES_DEFINITION = 'If the drift score is medium/high (above 0.1), we recommend you to check those features.\nA feature is considered risky if it is both in the top 40% of the most drifted features as well as the top 40% most important features in the original model.'
    MOST_DRIFTED_FEATURES = 'most_drifted_features'

    NUMBER_OF_DRIFTED_FEATURES = 20
    MOST_DRIFTED_FEATURES_DEFINITION = 'When the drift score is medium/high (above 0.1), this is the list of features that have been drifted the most, with their % of importance (max {0} features).'.format(NUMBER_OF_DRIFTED_FEATURES)
    MOST_IMPORTANT_FEATURES = 'most_important_features_in_deployed_model'
    MOST_IMPORTANT_FEATURES_DEFINTIION = 'Most important features in the deployed model, with their % of importance (max 20 features).'
    FEATURE_IMPORTANCE = 'feature_importance'

    ORIGIN_COLUMN = '__dku_row_origin__'  # name for the column that will contain the information from where the row is from (original test dataset or new dataframe)
    FROM_ORIGINAL = 'original'
    FROM_NEW = 'new'
    MIN_NUM_ROWS = 500
    MAX_NUM_ROW = 100000
    CUMULATIVE_PERCENTAGE_THRESHOLD = 90
    PREDICTION_TEST_SIZE = 100000
    SURROGATE_TARGET = "_dku_predicted_label_"

    REGRRSSION_TYPE = 'REGRESSION'
    CLASSIFICATION_TYPE = 'CLASSIFICATION'
    CLUSTERING_TYPE = 'CLUSTERING'
    DKU_CLASSIFICATION_TYPE = ['BINARY_CLASSIFICATION', 'MULTICLASS']


    FEAT_IMP_CUMULATIVE_PERCENTAGE_THRESHOLD = 95
    RISKIEST_FEATURES_RATIO_THRESHOLD = 0.65

    FEATURE = 'feature'
    IMPORTANCE = 'importance'
    CUMULATIVE_IMPORTANCE = 'cumulative_importance'
    RANK = 'rank'
    CLASS = 'class'
    PERCENTAGE = 'percentage'
    ORIGINAL_DATASET = 'original_dataset'
    NEW_DATASET = 'new_dataset'
    FUGACITY_RELATIVE_CHANGE_CLASSIF_LABEL = 'fugacity_relative_change_of_class_{0}'
    FUGACITY_RELATIVE_CHANGE_REGRESSION_LABEL = 'fugacity_relative_change_decile_{0}'
    FUGACITY_CLASSIF_LABEL = 'fugacity_class_{0}'


    @staticmethod
    def get_supported_metrics():
        return ModelDriftConstants.DRIFT_SCORE, ModelDriftConstants.FUGACITY, ModelDriftConstants.FEATURE_IMPORTANCE, ModelDriftConstants.RISKIEST_FEATURES