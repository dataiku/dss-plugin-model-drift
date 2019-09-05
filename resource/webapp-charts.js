function drawGauge(container, configuration){
    
    var barWidth, chart, chartInset, degToRad, repaintGauge,
        height, margin, numSections, padRad, percToDeg, percToRad, 
        percent, radius, sectionIndx, svg, totalPercent, width,
        valueText, formatValue, k;
        
      width = configuration['width']
      label = configuration['label']
      percentValue = configuration['percentValue'];
      firstSplitVal = 0.15;//configuration['firstSplitVal'];
      secondSplitVal = 0.2;//configuration['secondSplitVal'];
      thirdSplitVal = 0.15;configuration['thirdSplitVal']
      
      percent = percentValue;

      numSections = 1;
      sectionPerc = 1 / numSections / 2;
      padRad = 0.025;
      chartInset = 10;

      // Orientation of gauge:
      totalPercent = .75;

      el = d3.select(container);

      margin = {
        top: 30,
        right: 30,
        bottom: 30,
        left: 30
      };

      height = width;
      radius = Math.min(width, height) / 2;
      barWidth = 40 * width / 300;

      //Utility methods 

      percToDeg = function(perc) {
        return perc * 360;
      };

      percToRad = function(perc) {
        return degToRad(percToDeg(perc));
      };

      degToRad = function(deg) {
        return deg * Math.PI / 180;
      };

      // Create SVG element
      svg = el.append('svg').attr('width', width + margin.left + margin.right).attr('height', height + margin.top + margin.bottom);

      // Add layer for the panel
      chart = svg.append('g').attr('transform', "translate(" + ((width) / 2 + margin.left) + ", " + ((height + margin.top) / 2) + ")");


      chart.append('path').attr('class', "arc chart-first");
      chart.append('path').attr('class', "arc chart-second");
      chart.append('path').attr('class', "arc chart-third");

      valueText = chart.append("text")
                        .attr('id', "Value")
                        .attr("font-size",16)
                        .attr("text-anchor","middle")
                        .attr("dy",".5em")
                        .style("fill", '#000000');
      formatValue = d3.format('.02f');

      arc3 = d3.arc().outerRadius(radius - chartInset).innerRadius(radius - chartInset - barWidth)
      arc2 = d3.arc().outerRadius(radius - chartInset).innerRadius(radius - chartInset - barWidth)
      arc1 = d3.arc().outerRadius(radius - chartInset).innerRadius(radius - chartInset - barWidth)

      function repaintGauge() 
      {          
        perc = 0.5;
        var next_start = 0.75;
                    
        arcStartRad = percToRad(next_start);
        arcEndRad = arcStartRad + percToRad(firstSplitVal);
        next_start += firstSplitVal;
        arc1.startAngle(arcStartRad).endAngle(arcEndRad);

        arcStartRad = percToRad(next_start);
        arcEndRad = arcStartRad + percToRad(secondSplitVal);
        next_start += secondSplitVal;
        arc2.startAngle(arcStartRad + padRad).endAngle(arcEndRad);

        arcStartRad = percToRad(next_start);
        arcEndRad = arcStartRad + percToRad(thirdSplitVal);
        arc3.startAngle(arcStartRad + padRad).endAngle(arcEndRad);

        chart.select(".chart-first").attr('d', arc1);
        chart.select(".chart-second").attr('d', arc2);
        chart.select(".chart-third").attr('d', arc3);

      }
        chart.append("text")
             .text(label)
             .attr('id', "Name")
             .attr('x', 0)
             .attr('y', 30)
             .attr("text-anchor", "middle")
             .attr("font-size",15)
             .style("fill", "#000000");


      var Needle = (function() {

        //Helper function that returns the `d` value for moving the needle
        var recalcPointerPos = function(perc) {
          var centerX, centerY, leftX, leftY, rightX, rightY, thetaRad, topX, topY;
          thetaRad = percToRad(perc / 2);
          centerX = 0;
          centerY = 0;
          topX = centerX - this.len * Math.cos(thetaRad);
          topY = centerY - this.len * Math.sin(thetaRad);
          leftX = centerX - this.radius * Math.cos(thetaRad - Math.PI / 2);
          leftY = centerY - this.radius * Math.sin(thetaRad - Math.PI / 2);
          rightX = centerX - this.radius * Math.cos(thetaRad + Math.PI / 2);
          rightY = centerY - this.radius * Math.sin(thetaRad + Math.PI / 2);


            return "M " + leftX + " " + leftY + " L " + topX + " " + topY + " L " + rightX + " " + rightY;
        };

        function Needle(el) {
          this.el = el;
          this.len = width /2.8;
          this.radius = this.len / 9;
        }

        Needle.prototype.render = function() {
          this.el.append('circle').attr('class', 'needle-center').attr('cx', 0).attr('cy', 0).attr('r', this.radius);
          return this.el.append('path').attr('class', 'needle').attr('id', 'client-needle').attr('d', recalcPointerPos.call(this, 0));
        };

        Needle.prototype.moveTo = function(perc) {
          var self,
              oldValue = this.perc || 0;

          this.perc = perc;
          self = this;

          // Reset pointer position
        this.el.transition().delay(100).ease(d3.easeElastic).duration(200).select('.needle').tween('reset-progress', function() {
             var needle = d3.select(this);
            return function(percentOfPercent) {
              var progress = (1 - percentOfPercent) * oldValue;         
              repaintGauge(progress);
              return needle.attr('d', recalcPointerPos.call(self, progress));
            };
          });

          this.el.transition().delay(300).ease(d3.easeElastic).duration(1500).select('.needle').tween('progress', function() {
              var needle = d3.select(this);
            return function(percentOfPercent) {
              var progress = percentOfPercent * perc;
              repaintGauge(progress);
              var thetaRad = percToRad(progress / 2);
              var textX = - (self.len + 45) * Math.cos(thetaRad);
              var textY = - (self.len + 45) * Math.sin(thetaRad) + 10;

              valueText.text(formatValue(progress))
                .attr('transform', "translate("+textX+","+textY+")").attr('font-size',12);

              return needle.attr('d', recalcPointerPos.call(self, progress));
            };
          });
        };   

        return Needle;
      })();

      needle = new Needle(chart);
      needle.render();
      needle.moveTo(percent);

    };


function drawGauge2(container, configuration){
    
    var barWidth, chart, chartInset, degToRad, repaintGauge,
        height, margin, numSections, padRad, percToDeg, percToRad, 
        percent, radius, sectionIndx, svg, totalPercent, width,
        valueText, formatValue, k;
        
      width = configuration['width']
      label = configuration['label']
      percentValue = configuration['percentValue'];
      firstSplitVal = 0.025;//configuration['firstSplitVal'];
      secondSplitVal = 0.075;//configuration['secondSplitVal'];
      thirdSplitVal = 0.4;//configuration['thirdSplitVal']
      
      percent = percentValue;

      numSections = 1;
      sectionPerc = 1 / numSections / 2;
      padRad = 0.025;
      chartInset = 10;

      // Orientation of gauge:
      totalPercent = .75;

      el = d3.select(container);

      margin = {
        top: 30,
        right: 30,
        bottom: 30,
        left: 30
      };

      height = width;
      radius = Math.min(width, height) / 2;
      barWidth = 40 * width / 300;

      //Utility methods 

      percToDeg = function(perc) {
        return perc * 360;
      };

      percToRad = function(perc) {
        return degToRad(percToDeg(perc));
      };

      degToRad = function(deg) {
        return deg * Math.PI / 180;
      };

      // Create SVG element
      svg = el.append('svg').attr('width', width + margin.left + margin.right).attr('height', height + margin.top + margin.bottom);

      // Add layer for the panel
      chart = svg.append('g').attr('transform', "translate(" + ((width) / 2 + margin.left) + ", " + ((height + margin.top) / 2) + ")");


      chart.append('path').attr('class', "arc chart-first");
      chart.append('path').attr('class', "arc chart-second");
      chart.append('path').attr('class', "arc chart-third");

      valueText = chart.append("text")
                        .attr('id', "Value")
                        .attr("font-size",16)
                        .attr("text-anchor","middle")
                        .attr("dy",".5em")
                        .style("fill", '#000000');
      formatValue = d3.format('.02f');

      arc3 = d3.arc().outerRadius(radius - chartInset).innerRadius(radius - chartInset - barWidth)
      arc2 = d3.arc().outerRadius(radius - chartInset).innerRadius(radius - chartInset - barWidth)
      arc1 = d3.arc().outerRadius(radius - chartInset).innerRadius(radius - chartInset - barWidth)

      function repaintGauge2()
      {
        perc = 0.5;
        var next_start = 0.75;
                    
        arcStartRad = percToRad(next_start);
        arcEndRad = arcStartRad + percToRad(firstSplitVal);
        next_start += firstSplitVal;
        arc1.startAngle(arcStartRad).endAngle(arcEndRad);

        arcStartRad = percToRad(next_start);
        arcEndRad = arcStartRad + percToRad(secondSplitVal);
        next_start += secondSplitVal;
        arc2.startAngle(arcStartRad + padRad).endAngle(arcEndRad);

        arcStartRad = percToRad(next_start);
        arcEndRad = arcStartRad + percToRad(thirdSplitVal);
        arc3.startAngle(arcStartRad + padRad).endAngle(arcEndRad);

        chart.select(".chart-first").attr('d', arc1);
        chart.select(".chart-second").attr('d', arc2);
        chart.select(".chart-third").attr('d', arc3);

      }
        chart.append("text")
             .text(label)
             .attr('id', "Name")
             .attr('x', 0)
             .attr('y', 30)
             .attr("text-anchor", "middle")
             .attr("font-size",15)
             .style("fill", "#000000");


      var Needle = (function() {

        //Helper function that returns the `d` value for moving the needle
        var recalcPointerPos = function(perc) {
          var centerX, centerY, leftX, leftY, rightX, rightY, thetaRad, topX, topY;
          thetaRad = percToRad(perc / 2);
          centerX = 0;
          centerY = 0;
          topX = centerX - this.len * Math.cos(thetaRad);
          topY = centerY - this.len * Math.sin(thetaRad);
          leftX = centerX - this.radius * Math.cos(thetaRad - Math.PI / 2);
          leftY = centerY - this.radius * Math.sin(thetaRad - Math.PI / 2);
          rightX = centerX - this.radius * Math.cos(thetaRad + Math.PI / 2);
          rightY = centerY - this.radius * Math.sin(thetaRad + Math.PI / 2);


            return "M " + leftX + " " + leftY + " L " + topX + " " + topY + " L " + rightX + " " + rightY;
        };

        function Needle(el) {
          this.el = el;
          this.len = width /2.8;
          this.radius = this.len / 9;
        }

        Needle.prototype.render = function() {
          this.el.append('circle').attr('class', 'needle-center').attr('cx', 0).attr('cy', 0).attr('r', this.radius);
          return this.el.append('path').attr('class', 'needle').attr('id', 'client-needle').attr('d', recalcPointerPos.call(this, 0));
        };

        Needle.prototype.moveTo = function(perc) {
          console.warn('GAUGE 2', label); 
          var self,
              oldValue = this.perc || 0;

          this.perc = perc;
          self = this;

          // Reset pointer position
        this.el.transition().delay(100).ease(d3.easeElastic).duration(200).select('.needle').tween('reset-progress', function() {
             var needle = d3.select(this);
            return function(percentOfPercent) {
              var progress = (1 - percentOfPercent) * oldValue;         
              repaintGauge2(progress);
              return needle.attr('d', recalcPointerPos.call(self, progress));
            };
          });

          this.el.transition().delay(300).ease(d3.easeElastic).duration(1500).select('.needle').tween('progress', function() {
              var needle = d3.select(this);
            return function(percentOfPercent) {
              var progress = percentOfPercent * perc;
              repaintGauge2(progress);
              var thetaRad = percToRad(progress / 2);
              var textX = - (self.len + 45) * Math.cos(thetaRad);
              var textY = - (self.len + 45) * Math.sin(thetaRad) + 10;

              valueText.text(formatValue(progress))
                .attr('transform', "translate("+textX+","+textY+")").attr('font-size',12);

              return needle.attr('d', recalcPointerPos.call(self, progress));
            };
          });
        };   

        return Needle;
      })();

      needle = new Needle(chart);
      needle.render();
      needle.moveTo(percent);

    };