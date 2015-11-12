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

requestJSON("small_tree.json", function(response) {
	tree_json = JSON.parse(response);
	console.log(tree_json);


	// TODO: Allow user to load variable names
	// Thus far, hardcode variable names
	var var_names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'];
	// Likewise, hardcode label names
	var class_names = [0,1,2];

	// Set up basic parameters of the tree


	// determine node size and branch size

	var depth = tree_json.depth;
	finalkey = "lvl"+(depth-1);
	var spread = Math.pow(depth, 2);
	var node_margin = 20;
	var node_width = 50;
	var node_height = 50;
	var branch_height = node_height/2;
	var padding = node_width*0.1;

	// Load canvas element and context

	var canvas = document.getElementById('tree'),
	    context = canvas.getContext('2d');

	// Set width and height
	context.canvas.width = (node_width+node_margin)*(spread);
	context.canvas.height = (node_height+branch_height)*(depth+1);
	var midpoint = (node_width+node_margin)*(spread)/2.2;

	// Function to draw a node

	function drawNode(json, level, node, originx, originy, names) {
		key = "lvl" + level;
		n = json.levels[key][node];
		// draw rectangle

		context.strokeRect(originx, originy, node_width, node_height);

		// write name and rule

		if(!n.terminal) {
			context.fillStyle = "gray";
			context.font = "10px sans-serif";
			context.fillText(names[n.feature],originx + padding, originy + node_height/2.3);
			context.font = "11px monospace";
			context.fillText("<" + n.threshold.toFixed(2), originx + padding, originy + node_height/1.7);
			left = n.outcome[0].index;
			right = n.outcome[1].index;
			level_left = n.outcome[0].level;
			level_right = n.outcome[1].level;

			reverse_key = "lvl" + (depth-2-level)
			nodes_above = json.levels[reverse_key].length;
			
			names.splice(n.feature, 1);
			drawBranches(json, originx, originy, level_left, level_right, left, right, nodes_above, names);	
		} else {
			context.font = "bold 11px sans-serif";
			context.fillStyle = "aqua";
			context.fillRect(originx, originy, node_width, node_height);
			context.fillStyle = "black";
			context.fillText(class_names[n.outcome], originx + padding, originy + node_height/2);
			
		}

	}

	function drawBranches(json, nodex, nodey, level_left, level_right, left, right, nodes_above, names) {
		// Move from corner to the bottom center
		var startx = nodex + (node_width/2);
		var starty = nodey + (node_height);

		// Draw left line

		context.beginPath();
		context.moveTo(startx, starty);
		context.lineTo(startx - (node_width)*nodes_above, starty + branch_height);
		context.stroke();

		// Recursively draw another node
		xleft = startx - (node_width)*nodes_above - 0.5*node_width;
		yleft = starty + branch_height;
		drawNode(json, level_left, left, xleft, yleft, names);

		// Draw right line

		context.beginPath();
		context.moveTo(startx, starty);
		context.lineTo(startx + (node_width)*nodes_above, starty + branch_height);
		context.stroke();

		// Recursively draw another node
		xright = startx - 0.5*node_width + (node_width)*nodes_above; 
		yright = starty + branch_height;
		drawNode(json, level_right, right, xright, yright, names);

	}

	// Iterate over levels of the tree

	originx = midpoint;
	originy = 0;

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


