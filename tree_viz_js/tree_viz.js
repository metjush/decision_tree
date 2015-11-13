// Load JSON function

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

// Load canvas element and context

var canvas = document.getElementById('tree'),
		context = canvas.getContext('2d');

// Get JSON url and names from the input fields

// Form elements
var form = document.getElementById("setup_form");
var file = document.getElementById("json_file");
var variables = document.getElementById("json_names");
var classes = document.getElementById("json_classes");
var reset = document.getElementById("tree_reset");

var filename = "small_tree.json";
var var_names = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'];
var class_names = [0,1,2];

document.getElementById("json_submit").onclick = function(e) {
	e.preventDefault();
	filename = file.value;
	var_names = variables.value.split(" ");
	class_names = classes.value.split(" ");

	form.style.display = "none";
	reset.style.display = "block";

	// Draw the tree


	requestJSON(filename, function(response) {
		tree_json = JSON.parse(response);
		console.log(tree_json);


		// TODO: Allow user to load variable names
		// Thus far, hardcode variable names
		

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

		// Set width and height
		context.canvas.width = (node_width+node_margin)*(spread);
		context.canvas.height = (node_height+branch_height)*(depth+1);
		var midpoint = (node_width+node_margin)*(spread)/2.2;

		// Function to draw a node

		function drawNode(json, level, node, originx, originy, names) {
			key = "lvl" + level;
			n = json.levels[key][node];
			var vars = names;
			// draw rectangle

			context.strokeRect(originx, originy, node_width, node_height);

			// write name and rule

			if(!n.terminal) {
				context.fillStyle = "gray";
				context.font = "italic 10px sans-serif";
				context.fillText(vars[n.feature],originx + padding, originy + 15);
				context.font = "11px monospace";
				context.fillText("<" + n.threshold.toFixed(2), originx + padding, originy + 35);
				left = n.outcome[0].index;
				right = n.outcome[1].index;
				level_left = n.outcome[0].level;
				level_right = n.outcome[1].level;

				reverse_key = "lvl" + (depth-2-level)
				nodes_above = json.levels[reverse_key].length;
				
				vars.splice(n.feature, 1);
				drawBranches(json, originx, originy, level_left, level_right, left, right, nodes_above, vars);	
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
			var vars = names;

			// Draw left line

			context.beginPath();
			context.moveTo(startx, starty);
			context.lineTo(startx - (node_width)*nodes_above, starty + branch_height);
			context.stroke();

			// Recursively draw another node
			xleft = startx - (node_width)*nodes_above - 0.5*node_width;
			yleft = starty + branch_height;
			drawNode(json, level_left, left, xleft, yleft, vars);

			// Draw right line

			context.beginPath();
			context.moveTo(startx, starty);
			context.lineTo(startx + (node_width)*nodes_above, starty + branch_height);
			context.stroke();

			// Recursively draw another node
			xright = startx - 0.5*node_width + (node_width)*nodes_above; 
			yright = starty + branch_height;
			drawNode(json, level_right, right, xright, yright, vars);

		}

		// Recursion!

		originx = midpoint;
		originy = 0;

		drawNode(tree_json, 0, 0, originx, originy, var_names);


	});


};

reset.onclick = function(e) {
	e.preventDefault();
	form.style.display = "block";
	context.clearRect(0, 0, canvas.width, canvas.height);
	reset.style.display = "none";
}




