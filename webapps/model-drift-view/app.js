let webAppConfig = dataiku.getWebAppConfig();
let modelId = webAppConfig['modelId'];
let versionId = webAppConfig['versionId'];

dataiku.webappBackend.get('list-datasets')
    .then(data => {
            $.each(data.dataset_list, function(i, option) {
                $('#dataset-selector').append($('<option/>').attr("value", option.name).text(option.name));
            });
        }
    );

$('#run-button').click(function() {
    dataiku.webappMessages.clear();
    runAnalysis($('#run-button'));
});


function changeInputColor(input, value){
        $(input).removeClass();
        if (value < 0.1){
            $(input).addClass('low-risk');
        }
        else if(value >= 0.1 && value <= 0.5){
            $(input).addClass('medium-risk');
        }
        else{
            $(input).addClass('high-risk');
        }
    }

function runAnalysis($this) {
    markRunning(true);
    dataiku.webappBackend.get('get-drift-metrics', {'model_id': modelId, 'version_id': versionId, 'test_set': $("#dataset-selector").val()})
        .then(
            function(data) {
                $('#drift-score').text(data['drift_accuracy']);
                changeInputColor('#drift-score', data['drift_accuracy']);
                $('#error_message').html('');
                draw(data);
                $('.result-state').show();
                markRunning(false);
            }
        )
        .catch(error => {
            markRunning(false);
            dataiku.webappMessages.displayFatalError(error);
        });
}

function markRunning(running) {
    if (running) {
        $('.running-state').show();
        $('.notrunning-state').hide();
        $('.result-state').hide();
    } else {
        $('.running-state').hide();
        $('.notrunning-state').show();
    }
}

function draw(data) {
    drawFugacity(data['fugacity']);
    drawKDE(data['kde'], data['stat_metrics']);
    drawFeatureImportance(data['feature_importance']);
}

function drawFugacity(data) {
    $('#fugacity-score').html(json2table(data, 'table ml-table'));
}

function json2table(json, classes) {
    let cols = Object.keys(json[0]);
    let header = '';
    let body = '';
    classes = classes || '';

    function capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }

    cols.map(function(col) {
        header += '<th>' + capitalizeFirstLetter(col) + '</th>';
    });

    json.map(function(row) {
        body += '<tr>';
        cols.map(function(colName) {
            body += '<td>' + row[colName] + '</td>';
        });
        body += '</tr>';
    });
    return `<table class="${classes}"><thead><tr>${header}</tr></thead><tbody>${body}</tbody></table>`;
}

function drawKDE(data, statMetrics) {
    d3.select("#kde-chart").select("svg").remove();
    d3.select("#label-list").selectAll("option").remove();
    
    let margin = {top: 30, right: 30, bottom: 30, left: 50};
    let width = 460 - margin.left - margin.right;
    let height = 400 - margin.top - margin.bottom;

    // append the svg object to the body of the page
    let svg = d3.select("#kde-chart")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    // List of groups (here I have one group per column)
    let labels = Object.keys(data);

    // add the options to the button
    d3.select("#label-list")
        .selectAll('myOptions')
        .data(labels)
        .enter()
        .append("option")
        .text(d => d) // text showed in the menu
        .attr("value", d => d) // corresponding value returned by the button
        .property("selected", d => d === labels[1]);

    // add the x Axis
    let x = d3.scaleLinear()
        .domain([0,100])
        .range([0, width]);
    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x));

    t_test = statMetrics[labels[1]]['t_test'];
    power = statMetrics[labels[1]]['power'];
    $("#t-test-result").text(t_test);

    let density1 = data[labels[1]]['original'];
    let density2 = data[labels[1]]['new'];

    density1_array = density1.map(x=>x[1])
    density2_array = density2.map(x=>x[1])

    // first and last value of array must be zero otherwise the color fill will mess up
    density1[0] = [0,0];
    density2[0] = [0,0];
    density1[density1.length - 1] = [100,0];
    density2[density2.length - 1] = [100, 0];

    // add the y Axis
    let maxY = Math.max.apply(Math, density1_array.concat(density2_array));
    let y = d3.scaleLinear()
        .range([height, 0])
        .domain([0, maxY*1.1]);
    svg.append("g")
        .call(d3.axisLeft(y));

    // Plot the area
    let curve1 = svg.append("path")
        .attr("class", "mypath")
        .datum(density1)
        .attr("fill", "#2b67ff")
        .attr("opacity", ".6")
        .attr("stroke", "#000")
        .attr("stroke-width", 2)
        .attr("stroke-linejoin", "round")
        .attr("d", d3.line()
            .curve(d3.curveBasis)
            .x(d => x(d[0]))
            .y(d => y(d[1]))
        );

    // Plot the area
    let curve2 = svg.append("path")
        .attr("class", "mypath")
        .datum(density2)
        .attr("fill", "#ff832b")
        .attr("opacity", ".6")
        .attr("stroke", "#000")
        .attr("stroke-width", 2)
        .attr("stroke-linejoin", "round")
        .attr("d", d3.line()
            .curve(d3.curveBasis)
            .x(d => x(d[0]))
            .y(d => y(d[1]))
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

        t_test = statMetrics[labels[1]]['t_test'];
        power = statMetrics[labels[1]]['power'];
        $("#t-test").text('T-test p-value: ' + t_test);
        $("#power").text('Statistical power: '+ power);

        // recompute density estimation
        density1 = data[selectedGroup]['original'];
        density2 = data[selectedGroup]['new'];
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
            .attr("d",    d3.line()
            .curve(d3.curveBasis)
                .x(d => x(d[0]))
                .y(d => y(d[1]))
            );
        curve2
            .datum(density2)
            .transition()
            .duration(1000)
            .attr("d",    d3.line()
            .curve(d3.curveBasis)
                .x(d => x(d[0]))
                .y(d => y(d[1]))
            );
    }

    // Listen to the slider?
    d3.select("#label-list").on("change", function(d) {
        selectedGroup = this.value
        updateChart(selectedGroup)
    });
}

function drawFeatureImportance(data) {
    d3.select("#feat-imp-plot").select("svg").remove();
    //sort bars based on value
    data = data.sort(function (a, b) {
        return d3.descending(a['original_model'], b['original_model']);
    });

    // set the dimensions and margins of the graph
    let margin = {top: 20, right: 30, bottom: 40, left: 90};
    let width = 460 - margin.left - margin.right;
    let height = 400 - margin.top - margin.bottom;

    // append the svg object to the body of the page
    let svg = d3.select("#feat-imp-plot")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    let max_x_drift = data.reduce((max, data) => data['drift_model'] > max ? data['drift_model'] : max, data[0]['drift_model']);

    let color = d3.scaleLinear()
        .domain([0, max_x_drift])
        .range(['#33fa20','#ff0e00'])
        .interpolate(d3.interpolateHcl);


    let max_x = data.reduce((max, data) => data['original_model'] > max ? data['original_model'] : max, data[0]['original_model']);

    // Add X axis
    let x = d3.scaleLinear()
        .domain([0, max_x])
        .range([ 0, width]);
    svg.append("g")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(x))
        .selectAll("text")
        .attr("transform", "translate(-10,0)rotate(-45)")
        .style("text-anchor", "end");

    // Y axis
    let y = d3.scaleBand()
        .range([ 0, height ])
        .domain(data.map(d => d['feature']))
        .padding(.1);
    svg.append("g")
        .call(d3.axisLeft(y))

    //Bars
    svg.selectAll("myRect")
        .data(data)
        .enter()
        .append("rect")
        .attr("x", x(0) )
        .attr("y", d => y(d['feature']))
        .attr("width", d => x(d['original_model']))
        .attr("height", y.bandwidth() )
        .attr("fill", d => color(d['drift_model']))
        .attr("opacity", 0.8)
        .append("text");
}
