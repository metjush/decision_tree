// Load JSON

function requestJSON(file, response) {
	var xobj = new XMLHttpRequest();
	xobj.overrideMimeType("application/json");
	xobj.open('GET', file, true);
  xobj.onreadystatechange = function () {
  	if (xobj.readyState == 4 && xobj.status == "200") {
      response(xobj.responseText);
    }
  };
  xobj.send(null);
}

var tree_json = 0;

requestJSON("iris_json.json", function(response) {
	tree_json = JSON.parse(response);
	console.log(tree_json);


	// TODO: Allow user to load variable names
	// Thus far, hardcode variable names
	var var_names = ['sepal length', 'sepal width', 'petal length', 'petal width'];

	// Set up basic parameters of the tree

	// available real estate

	var avail_height = window.innerHeight;
	var avail_width = window.innerWidth;
	var midpoint = avail_width/2.2;

	// determine node size and branch size

	var depth = tree_json.depth;
	var spread = Math.pow(2,depth);
	var node_margin = 20;
	var node_width = avail_width/spread - node_margin;
	var node_height = avail_height/(3*depth);
	var branch_height = node_height/2;
	var padding = node_width*0.1;

	// Load canvas element and context

	var canvas = document.getElementById('exampleCanvas'),
	    context = canvas.getContext('2d');

	// Set width and height
	context.canvas.width = avail_width;
	context.canvas.height = avail_height;

	// Function to draw a node

	function drawNode(json, level, node, originx, originy, names) {
		key = "lvl" + level;
		n = json.levels[key][node];
		// level and position start from 0
		console.log(n);

		// draw rectangle

		context.strokeRect(originx, originy, node_width, node_height);

		// write name and rule

		if(!n.terminal) {
			context.fillText(names[n.feature],originx + padding, originy + 2*padding);
			context.fillText("<" + n.threshold.toFixed(2), originx + padding, originy + 5*padding);
			left = n.outcome[0].index;
			right = n.outcome[1].index;
			level_left = n.outcome[0].level;
			level_right = n.outcome[1].level;
			key_below = "lvl" + (level+1);
			nodes_below = json.levels[key_below].length;
			names.splice(n.feature, 1);
			drawBranches(json, originx, originy, level_left, level_right, left, right, nodes_below, names);	
		} else {
			context.fillText(n.outcome, originx + padding, originy + 2*padding);
		}

	}

	function drawBranches(json, nodex, nodey, level_left, level_right, left, right, nodes_below, names) {
		// Move from corner to the bottom center
		var startx = nodex + (node_width/2);
		var starty = nodey + (node_height);

		// Draw left line

		context.beginPath();
		context.moveTo(startx, starty);
		context.lineTo(startx - node_width - nodes_below*node_margin, starty + branch_height);
		context.stroke();

		// Recursively draw another node
		xleft = startx - 1.5*node_width - nodes_below*node_margin;
		yleft = starty + branch_height;
		console.log(level_left, left);
		console.log(startx, starty);
		drawNode(json, level_left, left, xleft, yleft, names);

		// Draw right line

		context.beginPath();
		context.moveTo(startx, starty);
		context.lineTo(startx + node_width + nodes_below*node_margin, starty + branch_height);
		context.stroke();

		// Recursively draw another node
		console.log(level_right, right);
		console.log(startx, starty);
		xright = startx + 0.5*node_width + nodes_below*node_margin; 
		yright = starty + branch_height;
		drawNode(json, level_right, right, xright, yright, names);

	}

	// Iterate over levels of the tree

	originx = midpoint;
	originy = 0;
	console.log(originx, originy);

	drawNode(tree_json, 0, 0, originx, originy, var_names);

	/*

	for (level = 0; level < depth; level++) {
		// Get nodes for this level
		key = "lvl" + level;
		nodes = tree_json.levels[key];

		// Get level width (number of nodes)
		lvl_width = nodes.length;

		// Get starting position for the first node
		

		// Iterate over nodes and draw them
		to_pop = [];
		
		for (node = 0; node < lvl_width; node++) {
			n = nodes[node];
			name = var_names[n.feature];
			to_pop.push(n.feature);
			node_coords = drawNode(level, node, name, n.threshold, originx, originy);
			if(!n.terminal) {
				drawBranches(level, node_coords[0], node_coords[1]);
			}
		}

		// Sort feature indices
		to_pop = to_pop.sort();

		// Remove used features from variable names
		for (i = to_pop.length - 1; i >= 0; i--) {
			var_names.splice(to_pop[i], 1);
		}


	}
	*/

});


