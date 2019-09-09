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
    var $this = $(this);
    run_analyse($this, webappMessages, draw);
    }                
)


function run_analyse($this, webappMessages, callback){
    $this.button('loading')
    var test_set = $("#dataset-list").val();    
    $.getJSON(getWebAppBackendUrl('get_drift_metrics'), {'model_id': model_id, 'test_set': test_set})
        .done(
            function(data){
                //location.reload() // reload the html to clean the error message if exist
                $this.button('reset');
                $('#drift-score').text(data['drift_accuracy']);                
                $("#t-test").text('Student t-test: ' + JSON.stringify(data['stat_metrics']['proba_1']));
                $('#error_message').html('');
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
    draw_kde(data['predictions']);
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


function draw_kde(data){
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


      // add the x Axis
      var x = d3.scaleLinear()
          .domain([0,100])
          .range([0, width]);
      svg.append("g")
          .attr("transform", "translate(0," + height + ")")
          .call(d3.axisBottom(x));
    

      // Compute kernel density estimation
      var kde = kernelDensityEstimator(kernelEpanechnikov(10), x.ticks(50))
      var density1 =  kde(data['original'])
      var density2 =  kde(data['new'])

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
      svg.append("path")
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
      svg.append("path")
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
    svg.append("text").attr("x", 300).attr("y", 10).text("Original test set").style("font-size", "15px").attr("alignment-baseline","middle")
    svg.append("text").attr("x", 300).attr("y", 40).text("New test set").style("font-size", "15px").attr("alignment-baseline","middle")
    // Add X axis label:
    svg.append("text")
      .attr("text-anchor", "end")
      .attr("x", width/2 + 70)
      .attr("y", height + 30)
      .attr("font-size", 12)
      .text(" Prediction probability (in %) for label 1");

    // Function to compute density
    function kernelDensityEstimator(kernel, X) {
      return function(V) {
        return X.map(function(x) {
          return [x, d3.mean(V, function(v) { return kernel(x - v); })];
        });
      };
    }
    function kernelEpanechnikov(k) {
      return function(v) {
        return Math.abs(v /= k) <= 1 ? 0.75 * (1 - v * v) / k : 0;
      };
    }
}


function draw_feat_imp(data){
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
        .style("font-size", "20px")
      
    
        
        var tipMouseover = function(d) {
          var html  = d["feature"];
          tooltip.html(html)
              .style("left", d + "px")
              .style("top", d  + "px")
            .transition()
              .duration(200) // ms
              .style("opacity", .9) // started as 0!
      };
    
        var tipMouseover2 = function(d){
            var html  = d["feature"];
            var matrix = this.getScreenCTM().translate(+ this.getAttribute("cx"), + this.getAttribute("cy"));
            tooltip.html(html)
                .style("left", d3.select(this).attr("cx")+ "px")     
                .style("top", d3.select(this).attr("cy") + "px")
                .transition()
                  .duration(200) // ms
                  .style("opacity", .9) // started as 0!
        };
    
      // Add X axis label:
      svg.append("text")
          .attr("text-anchor", "end")
          .attr("x", width/2 + margin.left + 50)
          .attr("y", height + margin.top + 20)
          .attr("font-size", 12)
          .text("Drift model feature importance ranking");

      // Y axis label:
      svg.append("text")
          .attr("text-anchor", "end")
          .attr("transform", "rotate(-90)")
          .attr("y", -margin.left + 20)
          .attr("x", -margin.top - height/2 + 120)
          .attr("font-size", 12)
          .text("Original model feature importance ranking");
    
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
          .attr("r", 6)
          .style("fill", "#69b3a2")
    .on("mouseover", tipMouseover2 )
    .on("mouseout", tipMouseout )
}