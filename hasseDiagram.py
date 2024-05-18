##############
#HasseDiagram
##############
import math
import random
import utils #requires
try:
	import tkinter as tk
except:
	tk = "tkinter"

class HasseDiagram:
	r'''
	A class that can produce latex/tikz code for the Hasse diagram of a poset or display the diagram in a window using tkinter.


	##########################################
	#Overview
	##########################################

	An instance of this class is attached to each instance of Poset. This class
	is used to produce latex code for a poset when Poset.latex() is called or
	to display a poset in a new window when Poset.show() is called. These functions
	are wrappers for HasseDiagram.latex() and HasseDiagram.tkinter().

	The constructor for this class takes keyword arguments that control how the
	Hasse diagram is drawn. These keyword arguments set the default options
	for that given instance of HasseDiagram. When calling latex() to produce latex code
	or tkinter() to draw the diagram in a tkinter window the same keyword arguments can
	be passed to control how the diagram is drawn during that particular operation.

	Options such as height, width, scale, etc. are values whose defaults can
	be overriden by passing keyword arguments  either to
	the constructor or the functions themselves.

	Other options such as loc_x, loc_y, nodeLabel or nodeDraw are functions.
	The default values for these functions are class methods.

	##########################################
	#Overriding function parameters
	##########################################

	Function parameters can be overriden in two ways. The first option is to
	make a function with the same signature as the default function and to pass that
	function as a keyword argument to the constructor or latex()/tkinter() when called.

	For example:

		def nodeLabel(this, i):
			return str(this.P.mobius(0, this.P[i]))

		#P is a Poset already constructed that has a minimum 0
		P.hasseDiagram.tkinter(nodeLabel = nodeLabel)

	The code above will show a Hasse Diagram of P with the elements labeled by
	the mobius values $\mu(0,p)$.

	When overriding function parameters the first argument is always the HasseDiagram
	instance. HasseDiagram has an attribute for each option described below as well
	as the following attributes:

		P - The poset to be drawn.

		in_tkinter - Boolean indicating whether tkinter() is being executed.

		in_latex - Boolean indicating whether latex() is being executed.

		canvas - While tkinter() is being executed this is the tkinter.Canvas
			object being drawn to.

	Note that any function parameters, such as nodeLabel, are set via
		this.nodeLabel = #provided function
	so if you intend to call these functions you must pass this as an argument via
		this.nodeLabel(this, i)
	The class methods remain unchanged of course, for example HasseDiagram.nodeLabel
	always refers to the default implementation.

	##############
	#Subclassing
	##############

	The second way to override a function parameter is via subclassing. This is more
	convenient if overriding several function parameters at once or if the computations
	are more involved. It is also useful for adding extra parameters. Any variables initialized
	in the constructor are saved at the beginning or latex() or tkinter(), overriden during
	execution of the function by any provided keyword arguments, and restored at the end of
	execution. The mobius example above can be accomplished by subclassing as
	follows:
		class MobiusHasseDiagram(HasseDiagram):

			def nodeLabel(this, i):
				zerohat = this.P.min()[0]
				return str(this.P.mobius(zerohat, this.P[i]))

		P.hasseDiagram = MobiusHasseDiagram(P)
		P.hasseDiagram.tkinter()

	To provide an option that changes what element the mobius value is computed
	from just set the value in the constructor.

		class MobiusHasseDiagram(HasseDiagram):

			def __init__(this, P, z = None, **kwargs):
				super().__init__(P, **kwargs)

				if z == None:
					this.z = this.P.min()[0] #z defaults to first minimal element
				else:
					this.z = z

			def nodeLabel(this, i):
				return str(this.P.mobius(this.z, this.P[i]))

		#P is a Poset with minimum 0
		P.hasseDiagram = MobiusHasseDiagram(P)
		P.hasseDiagram.tkinter() #labels are $\mu(0, x)$
		P.hasseDiagram.tkinter(z = P[0]) #labels are $\mu(P_0, x)$
		P.hasseDiagram.tkinter() #labels are $\mu(0, x)$

	Note you can pass a class to the Poset constructor to construct a poset with
	a hasseDiagram of that class.

	##########################################
	#Keyword arguments
	##########################################

	Options that affect both latex() and tkinter():

		width - Width of the diagram. When calling latex() this is the width
			in tikz units, for tkinter() the units are 1/10th of tkinter's units.

			The default value is 18.

		height - Height of the diagram, uses the same units as width.

			The default value is 30.

		labels - Whether to display labels for elements or just a filled circle.

			The default value is True.

		ptsize - No effect when labels is True, when labels is False this is the size
			of the circles shown for elements. When calling tkinter() this can
			be either a number or a string; if a string the last two characters
			are ignored.

			The default value is '2pt'.

		indices_for_nodes - If True HasseDiagram.nodeLabel is not called and the
			node text is the index of the element in the poset.
			If labels is False this argument has no effect.

			The default value is False.

		nodeLabel - A function that given this and an index returns a string to label
			the corresponding element by.

			The default value is HasseDiagram.nodeLabel.

		loc_x - A function that given this and an index returns the x-coordinate
			of the element in the diagram as a string. 0 represents the middle
			of the screen, positive values rightward and negative leftward.
			Returned value should lie in the range [-width/2, width/2].

			The default value is HasseDiagram.loc_x.

		loc_y - A function that given this and an index returns the y-coordinate
			of the element in the diagram as a string. 0 represents the bottom
			of the screen, positive values extend upward. Returned value should
			lie in the range [0, height].

			The default value is HasseDiagram.loc_y.

		jiggle
		jiggle_x
		jiggle_y  - Coordinates of all elements are perturbed by a random vector in
			the rectangle
				-jiggle-jiggle_x <= x <= jiggle+jiggle_x
				-jiggle-jiggle_y <= y <= jiggle+jiggle_y
			This can be useful if you want to prevent cover lines from successive ranks
			aligning to form the illusion of a line crossing between two ranks;
			or when drawing unranked posets if a line happens to cross over an
			element. The perturbation occurs in loc_x and loc_y so if these are
			overwritten and you want to preserve this behaviour add a line
			to the end of loc_x such as
				x = x+random.uniform(-jiggle-jiggle_x,jiggle+jiggle_x)

			The default values are 0.

	Options that affect only tkinter():

		scale - All coordinates are scaled by 10*scale.

			The default value is 1.

		padding - A border of this width is added around all sides of the diagram.
			This is affected by scale.

			The default value is 3.

		offset - Cover lines start above the bottom element and end below the top
			element, this controls the separation.

			The default value is 1.

		nodeDraw - When labels is False this function is called instead of placing
			anything for the node. The function is passed this and an index to
			the element to be drawn. nodeDraw should use the tkinter.Canvas
			object this.canvas to draw. The center of your diagram should be
			at the point
				x = float(this.loc_x(this,i))*float(this.scale) + float(this.scale)*float(this.width)/2 + float(this.padding)
				y = 2*float(this.padding)+float(this.height)*float(this.scale)-(float(this.loc_y(this,i))*float(this.scale) + float(this.padding))
			For larger diagrams make sure to increase height and width as well as offset.

			The default value is HasseDiagram.nodeDraw.

	Options that affect only latex():

		extra_packages - A string that when calling latex() is placed in the preamble.
			It should be used to include any extra packages or define commands
			needed to produce node labels. This has no effect when standalone is False.

			The default value is ''.

		nodescale - Each node is wrapped in '\\scalebox{'+nodescale+'}'.

			The default value is '1'.

		tikzscale - This is the scale parameter for the tikz environment, i.e. the
			tikz environment containing the figure begins
			'\\begin{tikzpicture}[scale='+tikzscale+']'.

			The default value is '1'.

		line_options - Tikz options to be included on every line drawn, i.e. lines
			will be written as '\\draw['+line_options+'](...'.

			The default value is ''.

		northsouth - If True lines are not drawn between nodes directly but from
			node.north to node.south which makes lines come together just beneath
			and above nodes. When False lines are drawn directly to nodes which
			makes lines directed towards the center of nodes.

			The default is True.

		lowsuffix - When this is nonempty lines will be drawn to node.lowsuffix instead of
			directly to nodes for the higher node in each cover. If northsouth
			is True this has no effect, '.south' is used for the low suffix.

			The default is ''.

		highsuffix - This is the suffix for the bottom node in each cover. If northsouth
			is True this has no effect, '.north' is used for the high suffix.

			The default is ''.

		decoration - A function that takes this and indices
			i and j representing a cover this.P[i]<this.P[j] to be drawn and
			returns a string of tikz line options to be included (along with any
			line options from line_options) on that draw command.

			The default value is HasseDiagram.decoration which
			returns an empty string.

		nodeName - A function that takes this and an index i representing an element
			whose node is to be drawn and returns the name of the node in tikz.
			This does not affect the image but is useful if you intend to edit
			the latex code and want the node names to be human readable.

			The default value HasseDiagram.nodeName returns str(i).

		standalone - When True a preamble is added to the beginning and
			'\\end{document}' is added to the end so that the returned string
			is a full latex document that can be compiled. Compiling requires
			the latex packages tikz (pgf) and preview. The resulting figure can be
			incorporated into another latex document with \includegraphics.

			When False only the code for the figure is returned; the return value
			begins with \begin{tikzpicture} and ends with \end{tikzpicture}.

			The default is False.
	'''
	def __init__(this, P, **kwargs):
		'''
		See HasseDiagram.
		'''
		this.P = P
		this.in_latex = False
		this.in_tkinter = False

		this.defaults = {
			'extra_packages':'',
			'nodescale':'1',
			'scale':'1',
			'tikzscale':'1',
			'line_options':'',
			'northsouth':True,
			'lowsuffix':'',
			'highsuffix':'',
			'labels': True,
			'ptsize': '2pt',
			'height': 30,
			'width': 18,
			'decoration': type(this).decoration,
			'loc_x': type(this).loc_x,
			'loc_y': type(this).loc_y,
			'nodeLabel': type(this).nodeLabel,
			'nodeName': type(this).nodeName,
			'indices_for_nodes': False,
			'jiggle': 0,
			'jiggle_x': 0,
			'jiggle_y': 0,
			'standalone': False,
			'padding': 3,
			'nodeDraw': type(this).nodeDraw,
			'offset': 1,
			'color':'black',
			}

		for (k,v) in this.defaults.items():
			if k in kwargs: this.__dict__[k] = kwargs[k]
			else: this.__dict__[k] = v

	def decoration(this,i,j):
		'''
		This is the default implementation of decoration, it returns an empty string.
		'''
		return ''

	def loc_x(this, i):
		'''
		This is the default implementation of loc_x.

		This spaces elements along each rank evenly. The length of a rank is the
		ratio of the logarithms of the size of the rank and the size of the largest rank.

		The return value is a string.
		'''
		len_P=len(this.P)
		rk = this.P.rank(i, True)
		if len(this.P.ranks[rk])==1: return '0'
		rkwidth=math.log(float(len(this.P.ranks[rk])))/math.log(float(this.maxrksize))*float(this.width)
		index=this.P.ranks[rk].index(i)
		ret = (float(index)/float(len(this.P.ranks[rk])-1))*rkwidth - (rkwidth/2.0)
		jiggle = this.jiggle_x + this.jiggle
		return str( ret + random.uniform(-jiggle, jiggle) )

	def loc_y(this,i):
		'''
		This is the default value of loc_y.

		This evenly spaces ranks.

		The return value is a string.
		'''
		rk = this.P.rank(i, True)
		try: #divide by zero when P is an antichain
			delta = float(this.height)/float(len(this.P.ranks)-1)
		except:
			delta = 1
		jiggle = this.jiggle_y + this.jiggle
		return str( rk*delta + random.uniform(-jiggle,jiggle) )

	def nodeLabel(this,i):
		'''
		This is the default implementation of nodeLabel.

		The ith element is returned cast to a string.
		'''
		return str(this.P.elements[i])

	def nodeName(this,i):
		'''
		This is the default implementation of nodeName.

		i is returned cast to a string.
		'''
		return str(i)

	def nodeDraw(this, i):
		'''
		This is the default implementation of nodeDraw.

		This draws a filled black circle of radius ptsize/2.
		'''
		ptsize = this.ptsize if type(this.ptsize)==int else float(this.ptsize[:-2])

		x = float(this.loc_x(this,i))*float(this.scale) + float(this.scale)*float(this.width)/2 + float(this.padding)
		y = 2*float(this.padding)+float(this.height)*float(this.scale)-(float(this.loc_y(this,i))*float(this.scale) + float(this.padding))

		this.canvas.create_oval(x-ptsize/2,y-ptsize/2,x+ptsize/2,y+ptsize/2, fill=this.color)
		return

	@utils.requires(tk)
	def tkinter(this, **kwargs):
		'''
		Opens a window using tkinter and draws the Hasse diagram.

		The keyword arguments are described in HasseDiagram.
		'''
		#save default parameters to restore aferwards
		defaults = this.__dict__.copy()
		#update parameters from kwargs
		this.__dict__.update(kwargs)
		this.in_tkinter = True

		this.maxrksize = max([len(r) for r in this.P.ranks])

		root = tk.Tk()
		root.title("Hasse diagram of "+(this.P.name if hasattr(this.P,"name") else "a poset"))
		this.scale = float(this.scale)*10
		this.padding = float(this.padding)*this.scale
		width = float(this.width)*this.scale
		height = float(this.height)*this.scale
		canvas = tk.Canvas(root, width=width+2*this.padding, height=height+2*this.padding)
		this.canvas = canvas
		canvas.pack(fill = "both", expand = True)
		for r in range(len(this.P.ranks)):
			for i in this.P.ranks[r]:
				x = float(this.loc_x(this,i))*this.scale + width/2 + this.padding
				y = 2*this.padding+height-(float(this.loc_y(this,i))*this.scale + this.padding)
				if not this.labels:
					this.nodeDraw(this, i)
				else:
					canvas.create_text(x,y,text=str(i) if this.indices_for_nodes else this.nodeLabel(this,i),anchor='c')
				if r == len(this.P.ranks)-1: continue
				for j in [r for r in this.P.ranks[r+1] if this.P.less(i,r,True)] if this.P.isRanked() else this.P.filter([i], indices = True, strict = True).min():
					xj = float(this.loc_x(this,j))*this.scale + width/2 + this.padding
					yj = 2*this.padding+height-(float(this.loc_y(this,j))*this.scale + this.padding)
					canvas.create_line(x,y-this.scale*this.offset,xj,yj+this.scale*this.offset,color=this.color)
		root.mainloop() #makes this function blocking so you can actually see the poset when ran in a script
		this.__dict__.update(defaults)

	def latex(this, **kwargs):
		'''
		Returns a string to depict the Hasse diagram in Latex.

		The keyword arguments are described in HasseDiagram.
		'''
		this.maxrksize = max([len(r) for r in this.P.ranks])
		defaults = this.__dict__.copy()
		this.__dict__.update(kwargs)
		this.in_latex = True

		if this.northsouth:
			this.lowsuffix = '.north'
			this.highsuffix = '.south'
		##############
		#write preamble
		##############
		ret=[]
		######
		#parameters
		#####
		ret.append('%')
		temp = []
		for k in this.defaults:
			v = this.__dict__[k]
			temp.append(k+'='+(v.__name__ if callable(v) else repr(v)))
		ret.append(','.join(temp))
		del temp
		######
		######
		ret.append('\n')
		if this.standalone:
			ret.append('\\documentclass{article}\n\\usepackage{tikz}\n')
			ret.append(this.extra_packages)
			ret.append('\n\\usepackage[psfixbb,graphics,tightpage,active]{preview}\n')
			ret.append('\\PreviewEnvironment{tikzpicture}\n\\usepackage[margin=0in]{geometry}\n')
			ret.append('\\begin{document}\n\\pagestyle{empty}\n')
		ret.append('\\begin{tikzpicture}\n')

		###############
		#write nodes for the poset elements
		###############
		if not this.labels:
			for rk in this.P.ranks:
				for r in rk:
					name=this.nodeName(this, r)
					ret.append('\\coordinate('+name+')at('+this.loc_x(this, r)+','+this.loc_y(this, r)+');\n')
					ret.append('\\fill[color='+this.color+']('+name+')circle('+this.ptsize+');\n')
		else:
			for rk in this.P.ranks:
				for r in rk:
					ret.append('\\node[color='+this.color+']('+this.nodeName(this, r)+')at('+this.loc_x(this, r)+','+this.loc_y(this, r)+')\n{')
					ret.append('\\scalebox{'+this.nodescale+"}{")
					ret.append(str(r) if this.indices_for_nodes else this.nodeLabel(this, r))
					ret.append('}};\n\n')

		#draw lines for covers
		if this.P.isRanked():
			for r in range(0,len(this.P.ranks)-1):
				for i in this.P.ranks[r]:
					for j in this.P.ranks[r+1]:
#			for r in range(0,len(ranks)-1):
#				for i in ranks[r]:
#					for j in ranks[r+1]:
						if this.P.less(i,j,True):
							options=this.decoration(this, i,j)+(','+this.line_options if this.line_options!='' else "")
							if len(options)>0: options='['+options+']'
							ret.append('\\draw[color='+this.color+']'+options+'('+this.nodeName(this, i)+this.lowsuffix+')--('+this.nodeName(this, j)+this.highsuffix+");\n")

		#for unranked version for each element we have to check all the higher length elements
		if not this.P.isRanked():
#			for r in range(0,len(this.P.ranks)-1):
#				for i in this.P.ranks[r]:
#					for s in this.P.ranks[r+1:]: #<--there's 2 colons this time
			for r in len(0,len(ranks)-1):
				for i in ranks[r]:
					uoi=[] #elements above i
					for s in ranks[r+1:]:
						for j in s:
							if this.P.less(i,j, True):
								uoi.append(j)
					covers=poset_min(uoi,lambda i,j: this.P.less(i,j, True))
					for j in covers:
						options=this.decoration(this, i,j)+(','+this.line_options if this.line_options!='' else "")
						if len(options)>0: options='['+options+']'
						ret.append('\\draw[color='+this.color+']'+options+'('+this.nodeName(this, i)+this.lowsuffix+')--('+this.nodeName(this, j)+this.highsuffix+");\n")
		ret.append('\\end{tikzpicture}')
		if this.standalone:
			ret.append('\n\\end{document}')

		this.__dict__.update(defaults)
		return ''.join(ret)


##############
#End HasseDiagram
##############
