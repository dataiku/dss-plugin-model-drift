⚠️ Starting with DSS version 10.0.0, this plugin is considered as "Legacy" and will be maintained only to fix critical issues. We recommend using the native feature Model Evaluation Store: https://doc.dataiku.com/dss/latest/python-api/model-evaluation-stores.html


# Model drift monitoring

Monitoring ML models in production is often a tedious task. You can apply a simply retraining strategy based on monitoring the model’s performance: if your AUC drops by a given percentage, retrain. Although accurate, this approach requires to obtain the ground truth for your preditctions, which is not always fast, and certainly not “real time”.

Instead of waiting for the ground truth, we propose to look at the recent data the model has had to score, and statistically compare it with the data on which the model was evaluated. If these datasets are too different, the model may need to be retrained.


## Scope of the plugin
This plugin offers a set of different DSS components to monitor input data drift (of a model):
* Model view: visualise the drift metrics and graph
* Recipe: compute feature drift of a deployed model
* Recipe: compute drift between two datasets
* Custom metric: retrieve the most recent drift metric


## Installation and requirements

Please see our [official plugin page](https://www.dataiku.com/product/plugins/model-drift-monitoring/) for installation.

## Changelog

****Version 3.1.5 (2022-07)**
* Misc:
  * Load js package locally to support offline DSS instances.

****Version 3.1.4 (2022-03)**
* Misc:
  * Add cloudpickle to code-env requirements.
  * Update existing packages version.


****Version 3.1.3 (2021-12)**
* Enhancement: 
  * Use feature importance from Tree-based regression models.
  * Use surrogate model for CalibratedClassifierCV. 

**Version 3.0.0 (2020-12)**
* Enhancement:
  * Add binomial test to check the reliability of drift score.
  * Improve model view's UI.

**Version 2.0.0 (2020-06)**
* New components: 
   * Recipe: Compute feature drift of a deployed model
   * Recipe: Compute drift between two datasets
   * Custom metric: Retrieve last drift metric
* Enhancement: 
   * Add support for regression algorithms and non tree-based algorithm
   * Add riskiest features information allowing users to immediately have the list of features that they need to be careful about (ie. features that are drifted the most and are important in the deloyed model)
   * Add support for partitioning
   * Add support for all types of train-test split (with/without cross-validation)
* Bug fixes:
   * Fix bug with boolean dtype handling that leads to mismatch prediction probability and weird categorical variable encoding.
   * Fix bug with Date dtype and python 3.

**Version 1.0.0 (2019-12)**

* Initial release
* Model view component: support for tree-based classification algorithms

You can log feature requests or issues on our [dedicated Github repository](https://github.com/dataiku/dss-plugin-model-drift/issues).

# License

The Model drift monitoring plugin is:

   Copyright (c) 2020 Dataiku SAS
   Licensed under the [MIT License](LICENSE.md).
