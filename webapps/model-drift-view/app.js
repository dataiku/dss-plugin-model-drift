let webAppConfig = dataiku.getWebAppConfig();
let model_id = webAppConfig['idOfTheModel'];
let model_version = webAppConfig['idOfTheVersion'];

console.warn('model_id:', model_id)
console.warn('model_version:', model_version)


$('#run_analyse').on('click', function(){
    var $this = $(this);
    run_analyse($this, webappMessages, draw);
    }                
)

function run_analyse($this, webappMessages, callback){
    $this.button('loading')
    var test_set = $("#test_set").val();
    $.getJSON(getWebAppBackendUrl('get_drift_metrics'), {'model_id': model_id, 'test_set': test_set})
        .done(
            function(data){
                //location.reload() // reload the html to clean the error message if exist
                $this.button('reset');
                console.warn('toto--->', data);
                $('#auc').text('Drift model AUC: '+data['drift_auc']);                
                $('#accuracy').text('Drift model accuracy: '+data['drift_accuracy']);                
                $('#anderson-test').text('Anderson test: '+data['stat_metrics']['and_test']);
                $('#ks-test').text('KS test: '+data['stat_metrics']['ks_test']);
                $('#t-test').text('Student t-test: '+data['stat_metrics']['t_test']);
                $("#original-feat-imp").text('Feature importance'+JSON.stringify(data['feature_importance']));
                $('#error_message').html('');
                callback(data['feature_importance']);
            }
        ).error(function(data){
            $this.button('reset');
            webappMessages.displayFatalError('Internal Server Error: ' + data.responseText);
            }
         ); 
};

//let svg = d3.select("#feat-imp-plot");


function getMinX(data) {
  return data.reduce((min, p) => p['drift_model'] < min ? p['drift_model'] : min, data[0]['drift_model']);
}

function getMaxX(data) {
  return data.reduce((max, p) => p['drift_model'] > max ? p['drift_model'] : max, data[0]['drift_model']);
}

function getMinY(data) {
  return data.reduce((min, p) => p['original_model'] < min ? p['original_model'] : min, data[0]['original_model']);
}

function getMaxY(data) {
  return data.reduce((max, p) => p['original_model'] > max ? p['original_model'] : max, data[0]['original_model']);
}


function draw(data){
    
    
      google.charts.load('current', {'packages':['gauge']});
      google.charts.setOnLoadCallback(drawChart);

      function drawChart() {

        var data = google.visualization.arrayToDataTable([
          ['Label', 'Value'],
          ['Auc', 0],
          ['Accuracy', 0]
        ]);

        var options = {
          width: 400, height: 120,
          redFrom: 0.8, redTo: 1,
          yellowFrom:0.6, yellowTo: 0.8,
          minorTicks: 5,
          min:0,
          max:1,
          animation:{easing: 'inAndOut'}
        };

        var chart = new google.visualization.Gauge(document.getElementById('auc'));

       
        chart.draw(data, options);
        setTimeout(function() {
          data.setValue(0, 1, 0.5);
          data.setValue(1, 1, 0.8);
            chart.draw(data, options);
        }, 500);
      }
    
        
    //svg.selectAll("*").remove();
    // set the dimensions and margins of the graph
    var values = Object.keys(data).map(function(key){
        return data[key];
    });
    
    var maxX = getMaxX(values);
    var maxY = getMaxY(values);    
    
    var margin = {top: 10, right: 30, bottom: 30, left: 60},
        width = 460 - margin.left - margin.right,
        height = 400 - margin.top - margin.bottom;

    // append the svg object to the body of the page
    var svg = d3.select("#feat-imp-plot").append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform",
              "translate(" + margin.left + "," + margin.top + ")");

    //Read the data
      // Add X axis
    var x = d3.scaleLinear()
        .domain([0, maxX])
        .range([ 0, width]);
      svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x));

      // Add Y axis
      var y = d3.scaleLinear()
        .domain([0, maxY])
        .range([ height, 0]);
      svg.append("g")
        .call(d3.axisLeft(y));
    
    
    var tooltip = d3.select("#feat-imp-plot")
        .append("div")
        .style("opacity", 0)
        .attr("class", "tooltip")
        .style("background-color", "white")
        .style("border", "solid")
        .style("border-width", "1px")
        .style("border-radius", "5px")
        .style("padding", "10px")     
      
        var tipMouseover = function(d) {
          var html  = d["feature"];
          tooltip.html(html)
              .style("left", (d3.event.pageX + 15) + "px")
              .style("top", (d3.event.pageY - 28) + "px")
            .transition()
              .duration(200) // ms
              .style("opacity", .9) // started as 0!
      };
    
      // tooltip mouseout event handler
      var tipMouseout = function(d) {
          tooltip.transition()
              .duration(300) // ms
              .style("opacity", 0); // don't care about position!
      };
    // Add dots
      svg.append('g')
        .selectAll("dot")
        .data(values)
        .enter()
        .append("circle")
          .attr("cx", function (d) { return x(d['drift_model']); } )
          .attr("cy", function (d) { return y(d['original_model']); } )
          .attr("r", 8)
          .style("fill", "#69b3a2")
    .on("mouseover", tipMouseover )
    .on("mouseout", tipMouseout )

}