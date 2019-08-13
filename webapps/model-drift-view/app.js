let webAppConfig = dataiku.getWebAppConfig();
let model_id = webAppConfig['idOfTheModel'];
let model_version = webAppConfig['idOfTheVersion'];

console.warn('model_id:', model_id)
console.warn('model_version:', model_version)

run_analyse

$('#run_analyse').on('click', function(){
    run_analyse();
    }                
)

function run_analyse(){
    var test_set = $("#test_set").val();
    console.warn('RUN ANALYSE WITH ', test_set)
    $.getJSON(getWebAppBackendUrl('get_drift_metrics'), {'model_id': model_id, 'test_set': test_set})
        .done(
            function(data){
                console.warn('toto--->', data);
                $('#auc').text('Drift model AUC: '+data['drift_auc']);                
                $('#accuracy').text('Drift model accuracy: '+data['drift_accuracy']);                
                $('#anderson-test').text('Anderson test: '+data['stat_metrics']['and_test']);
                $('#ks-test').text('KS test: '+data['stat_metrics']['ks_test']);
                $('#t-test').text('Student t-test: '+data['stat_metrics']['t_test']);
                $("#original-feat-imp").text('Original feature importance'+JSON.stringify(data['feature_importance']['original_model']));
                $("#drift-feat-imp").text('Drift feature importance'+JSON.stringify(data['feature_importance']['drift_model']));
            }
        ); 
    
}


