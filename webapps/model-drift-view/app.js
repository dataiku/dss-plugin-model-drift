let webAppConfig = dataiku.getWebAppConfig();
let model_id = webAppConfig['idOfTheModel'];
let model_version = webAppConfig['idOfTheVersion']; 

console.warn('model_id:', model_id)
console.warn('model_version:', model_version)

$.getJSON(getWebAppBackendUrl('get_dataset_list'))
    .done( 
        function(data){
            var dataset_list = data['dataset_list'];
            $.each(dataset_list, function(i, option) {
                $('#dataset-list').append($('<option/>').attr("value", option.name).text(option.name));
            });
        }
);


$('#run_analyse').on('click', function(){
    sessionStorage.setItem("reloading", "true");
    localStorage.setItem("test_set", $("#dataset-list").val());
    $this = $(this);
    location.reload();
    }                
)

window.onload = function(){
    var reloading = sessionStorage.getItem("reloading");
    if (reloading){
        sessionStorage.removeItem("reloading");
        $this = $('#run_analyse');
        run_analyse($this, webappMessages, draw);
    }
}

function run_analyse($this, webappMessages, callback){
    var test_set = localStorage.getItem("test_set");
    document.getElementById("dataset-list").value = test_set;
    $this.button('loading')
    $.getJSON(getWebAppBackendUrl('get_drift_metrics'), {'model_id': model_id, 'model_version': model_version,'test_set': test_set}) // TODO: send model version too
        .done( 
            function(data){
                $this.button('reset');
                $('#drift-score').text(data['drift_accuracy']);                
                $('#error_message').html('');
                var label_list = data['label_list'];
                callback(data);
                document.getElementById('info-panel').style.display = "block";
            }
        ).fail(function(data){
            $this.button('reset');
            webappMessages.displayFatalError('Internal Server Error: ' + data.responseText);
            }
         ); 
};


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
    draw_fugacity(data['fugacity']);
    console.warn(data);
    draw_kde(data['kde'], data['stat_metrics']);
    draw_feat_imp(data['feature_importance']);
}


function draw_fugacity(data){
    document.getElementById('fugacity-score').innerHTML = json2table(data, 'table');
}

function json2table(json, classes){
    
    
    var cols = Object.keys(json[0]);
  
    var headerRow = '';
    var bodyRows = '';

   classes = classes || '';

      function capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
      }

      cols.map(function(col) {
        headerRow += '<th>' + capitalizeFirstLetter(col) + '</th>';
      });

      json.map(function(row) {
        bodyRows += '<tr>';

        cols.map(function(colName) {
          bodyRows += '<td>' + row[colName] + '</td>';
        })

        bodyRows += '</tr>';
      });

      return '<table class="' +
             classes +
             '"><thead><tr>' +
             headerRow +
             '</tr></thead><tbody>' +
             bodyRows +
             '</tbody></table>';
    }


function draw_kde(data, data_stats){
    var margin = {top: 30, right: 30, bottom: 30, left: 50};
    var width = 460 - margin.left - margin.right;
    var height = 400 - margin.top - margin.bottom;

    // append the svg object to the body of the page
    var svg = d3.select("#kde-chart")
      .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform",
              "translate(" + margin.left + "," + margin.top + ")");

    // List of groups (here I have one group per column)
    var label_list = Object.keys(data);

    // add the options to the button
    d3.select("#label-list")
        .selectAll('myOptions')
        .data(label_list)
        .enter()
        .append('option')
        .text(function (d) { return d; }) // text showed in the menu
        .attr("value", function (d) { return d; }) // corresponding value returned by the button
        .property("selected", function(d){ return d === label_list[1]; })

    
    // add the x Axis
    var x = d3.scaleLinear()
        .domain([0,100])
        .range([0, width]);
    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x));
    
    t_test = data_stats[label_list[1]]['t_test'];
    power = data_stats[label_list[1]]['power'];
    kde_text = 'The density chart illustrates how the original test dataset predicted probability distribution differs from that of the selected dataset. The density functions show the probability density of predicted rows in the test dataset (as positive) vs predicted rows in the selected dataset (as positive). A highly drifted model fully separates the density functions. <br> <br> The t-test aims at determining whether there is a significant difference between the mean of the two probability predicted distributions on the test dataset and on the selected dataset. Usually a p-value smaller than 0.005 means that the null hypothesis (the two populations have identical average values) is rejected. Here the p-value is ' + t_test
    $("#kde-explain").text(kde_text);

    var density1 =  data[label_list[1]]['original'];
    var density2 =  data[label_list[1]]['new'];

    density1_array = density1.map(x=>x[1])
    density2_array = density2.map(x=>x[1])

    // first and last value of array must be zero otherwise the color fill will mess up
    density1[0] = [0,0];
    density2[0] = [0,0];
    density1[density1.length - 1] = [100,0];
    density2[density2.length - 1] = [100, 0];
    
    // add the y Axis
    var maxY = Math.max.apply(Math, density1_array.concat(density2_array));
    var y = d3.scaleLinear()
            .range([height, 0])
            .domain([0, maxY*1.1]);
    svg.append("g")
      .call(d3.axisLeft(y));

    
    // Plot the area
    var curve1 = svg.append("path")
      .attr("class", "mypath")
      .datum(density1)
      .attr("fill", "#2b67ff")
      .attr("opacity", ".6")
      .attr("stroke", "#000")
      .attr("stroke-width", 2)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return x(d[0]); })
          .y(function(d) { return y(d[1]); })
      );

    
    // Plot the area
    var curve2 = svg.append("path")
      .attr("class", "mypath")
      .datum(density2)
      .attr("fill", "#ff832b")
      .attr("opacity", ".6")
      .attr("stroke", "#000")
      .attr("stroke-width", 2)
      .attr("stroke-linejoin", "round")
      .attr("d",  d3.line()
        .curve(d3.curveBasis)
          .x(function(d) { return x(d[0]); })
          .y(function(d) { return y(d[1]); })
      );
 

    // Handmade legend
    svg.append("circle").attr("cx",280).attr("cy",10).attr("r", 6).style("fill", "#2b67ff")
    svg.append("circle").attr("cx",280).attr("cy",40).attr("r", 6).style("fill", "#ff832b")
    svg.append("text").attr("x", 300).attr("y", 10).text("Original set").style("font-size", "15px").attr("alignment-baseline","middle")
    svg.append("text").attr("x", 300).attr("y", 40).text("New set").style("font-size", "15px").attr("alignment-baseline","middle")
    // Add X axis label:
    svg.append("text")
      .attr("text-anchor", "end")
      .attr("x", width/2 + 90)
      .attr("y", height + 29)
      .attr("font-size", 12)
      .text(" Probability predicted (in %) distribution ");

      // A function that update the chart when slider is moved?
    function updateChart(selectedGroup) {
          
          t_test = data_stats[label_list[1]]['t_test'];
          power = data_stats[label_list[1]]['power'];
          $("#t-test").text('T-test p-value: ' + t_test);
          $("#power").text('Statistical power: '+ power);          
          
        // recompute density estimation        
        //kde = kernelDensityEstimator(kernelEpanechnikov(7), x.ticks(20));
        density1 =  data[selectedGroup]['original'];
        density2 =  data[selectedGroup]['new'];
        // first and last value of array must be zero otherwise the color fill will mess up
        density1[0] = [0,0];
        density2[0] = [0,0];
        density1[density1.length - 1] = [100,0];
        density2[density2.length - 1] = [100, 0];
        density1_array = density1.map(x=>x[1])
        density2_array = density2.map(x=>x[1])
        // add the y Axis
        maxY = Math.max.apply(Math, density1_array.concat(density2_array));
        y.domain([0, maxY*1.1]);

        // update the chart
        curve1
          .datum(density1)
          .transition()
          .duration(1000)
          .attr("d",  d3.line()
            .curve(d3.curveBasis)
              .x(function(d) { return x(d[0]); })
              .y(function(d) { return y(d[1]); })
          );
          
        curve2
          .datum(density2)
          .transition()
          .duration(1000)
          .attr("d",  d3.line()
            .curve(d3.curveBasis)
              .x(function(d) { return x(d[0]); })
              .y(function(d) { return y(d[1]); })
          );
      }

      // Listen to the slider?
      d3.select("#label-list").on("change", function(d){
        selectedGroup = this.value
        updateChart(selectedGroup)
      })    
}


function draw_feat_imp(data){
    //sort bars based on value
    data = data.sort(function (a, b) {
        return d3.descending(a['original_model'], b['original_model']);
    })
    
    // set the dimensions and margins of the graph
    var margin = {top: 20, right: 30, bottom: 40, left: 90},
        width = 460 - margin.left - margin.right,
        height = 400 - margin.top - margin.bottom;

    // append the svg object to the body of the page
    var svg = d3.select("#feat-imp-plot")
      .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
      .append("g")
        .attr("transform",
              "translate(" + margin.left + "," + margin.top + ")");
    
    var max_x_drift = data.reduce((max, data) => data['drift_model'] > max ? data['drift_model'] : max, data[0]['drift_model']);  

    var color = d3.scaleLinear()
        .domain([0, max_x_drift])
        .range(['#33fa20','#ff0e00'])
        .interpolate(d3.interpolateHcl); 
    
    
    var max_x = data.reduce((max, data) => data['original_model'] > max ? data['original_model'] : max, data[0]['original_model']);  

    // Add X axis
    var x = d3.scaleLinear()
        .domain([0, max_x])
        .range([ 0, width]);
    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x))
        .selectAll("text")
          .attr("transform", "translate(-10,0)rotate(-45)")
          .style("text-anchor", "end");

    // Y axis
    var y = d3.scaleBand()
        .range([ 0, height ])
        .domain(data.map(function(d) { return d['feature']; }))
        .padding(.1);
    svg.append("g")
        .call(d3.axisLeft(y))

    //Bars
    svg.selectAll("myRect")
        .data(data)
        .enter()
        .append("rect")
        .attr("x", x(0) )
        .attr("y", function(d) { return y(d['feature']); })
        .attr("width", function(d) { return x(d['original_model']); })
        .attr("height", y.bandwidth() )
        .attr("fill", function(d) {return color(d['drift_model'])})
        .attr("opacity", 0.8)
        .append("text") 
}
