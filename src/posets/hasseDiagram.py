import random
import math
from .utils import *

try:
	import tkinter as tk
except:
	tk = "tkinter"

class HasseDiagram:
	r'''
	@is_section@
	A class that can produce latex/tikz code for the Hasse diagram of a poset or display the diagram in a window using tkinter.


	\textbf{\large{Overview}}

	An instance of this class is attached to each instance of \verb|Poset|. This class
	is used to produce latex code for a poset when \verb|Poset.latex()| is called or
	to display a poset in a new window when \verb|Poset.show()| is called. These functions
	are wrappers for \verb|HasseDiagram.latex()| and \verb|HasseDiagram.tkinter()|.

	The constructor for this class takes keyword arguments that control how the
	Hasse diagram is drawn. These keyword arguments set the default options
	for that given instance of \verb|HasseDiagram|. When calling \verb|latex()|
	to produce latex code
	or \verb|tkinter()| to draw the diagram in a tkinter window the same keyword arguments can
	be passed to control how the diagram is drawn during that particular operation.

	Options such as \verb|height|, \verb|width|, \verb|scale|, etc. are values whose defaults can
	be overriden by passing keyword arguments  either to
	the constructor or the functions themselves.

	Other options such as \verb|loc_x|, \verb|loc_y|, \verb|nodeLabel| or \verb|nodeDraw| are functions.
	The default values for these functions are class methods.

	\textbf{\large Overriding function parameters}

	Function parameters can be overriden in two ways. The first option is to
	make a function with the same signature as the default function and to pass that
	function as a keyword argument to the constructor or \verb|latex()|/\verb|tkinter()| when called.

	For example:
	\begin{center}
		\begin{verbatim}def nodeLabel(this, i):
			return str(this.P.mobius(0, this.P[i]))

		#P is a Poset already constructed that has a minimum 0
		P.hasseDiagram.tkinter(nodeLabel = nodeLabel)\end{verbatim}
	\end{center}

	The code above will show a Hasse Diagram of \verb|P| with the elements labeled by
	the M\"obius values $\mu(0,p)$.

	When overriding function parameters the first argument is always the HasseDiagram
	instance. HasseDiagram has an attribute for each option described below as well
	as the following attributes:

		\begin{itemize}

			\item{\verb|P| -- The poset to be drawn.}

			\item{\verb|in_tkinter| -- Boolean indicating whether \verb|tkinter()| is being executed.}

			\item{\verb|in_latex| -- Boolean indicating whether \verb|latex()| is being executed.}

			\item{\verb|canvas| -- While \verb|tkinter()| is being executed this is the \verb|tkinter.Canvas|
				object being drawn to.}
		\end{itemize}

	Note that any function parameters, such as \verb|nodeLabel|, are set via
		\[\verb|this.nodeLabel = #provided function|\]
	so if you intend to call these functions you must pass this as an argument via
		\[\verb|this.nodeLabel(this, i)|\]
	The class methods remain unchanged of course, for example \verb|HasseDiagram.nodeLabel|
	always refers to the default implementation.

	\textbf{\large Subclassing}

	The second way to override a function parameter is via subclassing. This is more
	convenient if overriding several function parameters at once or if the computations
	are more involved. It is also useful for adding extra parameters. Any variables initialized
	in the constructor are saved at the beginning or \verb|latex()| or \verb|tkinter()|, overriden during
	execution of the function by any provided keyword arguments, and restored at the end of
	execution. The M\"obius example above can be accomplished by subclassing as
	follows:
	\begin{center}
	\begin{BVerbatim}
	class MobiusHasseDiagram(HasseDiagram):

		def nodeLabel(this, i):
			zerohat = this.P.min()[0]
			return str(this.P.mobius(zerohat, this.P[i]))

	P.hasseDiagram = MobiusHasseDiagram(P)
	P.hasseDiagram.tkinter()
	\end{BVerbatim}
	\end{center}

	To provide an option that changes what element the M\"obius value is computed
	from just set the value in the constructor.
		\begin{verbatim}class MobiusHasseDiagram(HasseDiagram):

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
		P.hasseDiagram.tkinter() #labels are $\mu(0, x)$\end{verbatim}

	Note you can pass a class to the \verb|Poset| constructor to construct a poset with
	a \verb|hasseDiagram| of that class.

	\textbf{\large Keyword arguments}

	Options that affect both \verb|latex()| and \verb|tkinter()|:

		\begin{itemize}

		\item[]{\verb|width| -- Width of the diagram. When calling \verb|latex()| this is the width
			in tikz units, for \verb|tkinter()| the units are $\frac{1}{10}$th of tkinter's units.

			The default value is 18.
		}

		\item[]{\verb|height| -- Height of the diagram, uses the same units as width.

			The default value is 30.
		}

		\item[]{\verb|labels| -- Whether to display labels for elements or just a filled circle.

			The default value is \verb|True|.
		}

		\item[]{\verb|ptsize| -- No effect when \verb|labels| is \verb|True|, when \verb|labels| is \verb|False| this is the size
			of the circles shown for elements. When calling \verb|tkinter()| this can
			be either a number or a string; if a string the last two characters
			are ignored. When calling \verb|latex()| this should be a string and
			include units.

			The default value is '2pt'.
		}

		\item[]{\verb|indices_for_nodes| -- If \verb|True| then'
			\verb|HasseDiagram.nodeLabel| is not called and the
			node text is the index of the element in the poset.
			If \verb|labels| is \verb|False| this argument has no effect.

			The default value is \verb|False|.
		}

		\item[]{\verb|nodeLabel| -- A function that given this and an index returns a string to label
			the corresponding element by.

			The default value is \verb|HasseDiagram.nodeLabel|.
		}

		\item[]{\verb|loc_x| -- A function that given this and an index returns the $x$-coordinate
			of the element in the diagram as a string. 0 represents the middle
			of the screen, positive values rightward and negative leftward.
			Returned value should lie in the range $[-\verb|width|/2, \verb|width|/2]$.

			The default value is \verb|HasseDiagram.loc_x|.
		}

		\item[]{\verb|loc_y| -- A function that given this and an index returns the $y$-coordinate
			of the element in the diagram as a string. 0 represents the bottom
			of the screen, positive values extend upward. Returned value should
			lie in the range $[0, \verb|height|]$.

			The default value is \verb|HasseDiagram.loc_y|.
		}

		\item[]{\verb|jiggle|
		\verb|jiggle_x|
		\verb|jiggle_y| -- Coordinates of all elements are perturbed by a random vector in
			the rectangle
				\begin{center}
				$-\verb|jiggle|-\verb|jiggle_x| \le x \le \verb|jiggle|+\verb|jiggle_x|$\\
				$-\verb|jiggle|-\verb|jiggle_y| \le y \le \verb|jiggle|+\verb|jiggle_y|$
				\end{center}
			This can be useful if you want to prevent cover lines from successive ranks
			aligning to form the illusion of a line crossing between two ranks;
			or when drawing unranked posets if a line happens to cross over an
			element. The perturbation occurs in \verb|loc_x| and \verb|loc_y| so if these are
			overwritten and you want to preserve this behaviour add a line
			to the end of \verb|loc_x| such as
				\begin{center}
				\verb|x = x+random.uniform(-jiggle-jiggle_x,jiggle+jiggle_x)|
				\end{center}

			The default values are 0.
		}
	\end{itemize}

	Options that affect only \verb|tkinter()|:

		\begin{itemize}
		\item[]{\verb|scale| -- All coordinates are scaled by $10*\verb|scale|$.

			The default value is 1.
		}

		\item[]{\verb|padding| -- A border of this width is added around all sides of the diagram.
			This is affected by \verb|scale|.

			The default value is 3.
		}

		\item[]{\verb|offset| -- Cover lines start above the bottom element and end below the top
			element, this controls the separation.

			The default value is 1.
		}

		\item[]{\verb|nodeDraw| -- When labels is \verb|False| this function is called instead of placing
			anything for the node. The function is passed \verb|this| and an index to
			the element to be drawn. \verb|nodeDraw| should use the \verb|tkinter.Canvas|
			object \verb|this.canvas| to draw. The center of your diagram should be
			at the point
				\begin{center}
	\begin{BVerbatim}
	x = float(this.loc_x(this,i)) * float(this.scale) + float(this.scale) * \
	float(this.width)/2 + float(this.padding)

	y = 2 * float(this.padding) + float(this.height) * \
	float(this.scale) - (float(this.loc_y(this,i)) * float(this.scale) + \
	float(this.padding))
	\end{BVerbatim}
				\end{center}
			For larger diagrams make sure to increase \verb|height| and \verb|width| as well as \verb|offset|.

			The default value is \verb|HasseDiagram.nodeDraw|.
		}
	\end{itemize}

	Options that affect only \verb|latex()|:

	\begin{itemize}
		\item[]{\verb|extra_packages| -- A string that when calling \verb|latex()| is placed in the preamble.
			It should be used to include any extra packages or define commands
			needed to produce node labels. This has no effect when standalone is \verb|False|.

			The default value is \verb|''|.
		}

		\item[]{\verb|nodescale| -- Each node is wrapped in \verb|'\\scalebox{'+nodescale+'}'|.

			The default value is \verb|'1'|.
		}

		item[]{\verb|tikzscale| -- This is the scale parameter for the tikz environment, i.e. the
			tikz environment containing the figure begins
			\begin{center}
				\verb|'\\begin{tikzpicture}[scale='+tikzscale+']'|
			\end{center}

			The default value is \verb|'1'|.
		}

		\item[]{\verb|line_options| -- Tikz options to be included on lines drawn, i.e. lines
			will be written as
			\begin{verbatim}'\\draw['+line_options+'](...'\end{verbatim}
			The value for
			\verb|line_options| can be either a string or a function; when it is
			a string the same options are placed on every line and when the value
			is a function it is passed \verb|this|, the \verb|HasseDiagram| object,
			\verb|i|, the index to the element at the bottom of the cover
			and \verb|j|, the index to the element at the top of the cover.

			The default value is \verb|''|.
		}

		\item[]{\verb|node_options| -- Tikz options to be included on nodes drawn,
			i.e. nodes will be written as
			\begin{verbatim}'\\node['+node_options_'](...'\end{verbatim}
			Just as with \verb|line_options| the value for \verb|node_options| can
			be either a string or a function; if it is a function it is passed
			\verb|this|, the \verb|HasseDiagram| object, and \verb|i|, the
			index to the element being drawn.
		}

		\item[]{\verb|northsouth| -- If \verb|True| lines are not drawn between nodes directly but from
			\verb|node.north| to \verb|node.south| which makes lines come together just beneath
			and above nodes. When \verb|False| lines are drawn directly to nodes which
			makes lines directed towards the center of nodes.

			The default is \verb|True|.
		}

		\item[]{\verb|lowsuffix| -- When this is nonempty lines will be drawn to \verb|node.lowsuffix| instead of
			directly to nodes for the higher node in each cover. If \verb|northsouth|
			is \verb|True| this has no effect, \verb|'.south'| is used for the low suffix.

			The default is \verb|''|.
		}

		\item[]{\verb|highsuffix| -- This is the suffix for the bottom node in each cover. If \verb|northsouth|
			is \verb|True| this has no effect, \verb|'.north'| is used for the high suffix.

			The default is \verb|''|.
		}

		\item[]{\verb|decoration| -- A function that takes \verb|this| and indices
			\verb|i| and \verb|j| representing a cover \verb|this.P[i]<this.P[j]| to be drawn and
			returns a string of tikz line options to be included (along with any
			line options from \verb|line_options|) on that draw command.

			The default value is \verb|HasseDiagram.decoration| which
			returns an empty string.
		}

		\item[]{\verb|nodeName| -- A function that takes \verb|this| and an index \verb|i| representing an element
			whose node is to be drawn and returns the name of the node in tikz.
			This does not affect the image but is useful if you intend to edit
			the latex code and want the node names to be human readable.

			The default value \verb|HasseDiagram.nodeName| returns \verb|str(i)|.
		}

		\item[]{\verb|standalone| -- When \verb|True| a preamble is added to the beginning and
			\verb|'\\end{document}'| is added to the end so that the returned string
			is a full latex document that can be compiled. Compiling requires
			the latex packages tikz (pgf) and preview. The resulting figure can be
			incorporated into another latex document with \verb|\includegraphics|.

			When \verb|False| only the code for the figure is returned; the return value
			begins with \verb|\begin{tikzpicture}| and ends with \verb|\end{tikzpicture}|.

			The default is \verb|False|.
		}
	\end{itemize}
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
			'node_options': type(this).node_options,
			'line_options': type(this).line_options,
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

		this.validate()

	def decoration(this,i,j):
		r'''
		This is the default implementation of \verb|decoration|, it returns an empty string.
		'''
		return ''

	def line_options(this,i,j):
		r'''
		This is the default implementation of \verb|line_options|, it returns an empty string.
		'''
		return ''
	def node_options(this,i):
		r'''
		This is the default implementation of \verb|node_options|, it returns an empty string.
		'''
		return ''
	def loc_x(this, i):
		r'''
		This is the default implementation of \verb|loc_x|.

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
		r'''
		This is the default value of \verb|loc_y|.

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
		r'''
		This is the default implementation of \verb|nodeLabel|.

		The $i$th element is returned cast to a string.
		'''
		return str(this.P.elements[i])

	def nodeName(this,i):
		r'''
		This is the default implementation of \verb|nodeName|.

		$i$ is returned cast to a string.
		'''
		return str(i)

	def nodeDraw(this, i):
		r'''
		This is the default implementation of \verb|nodeDraw|.

		This draws a filled black circle of radius $\verb|ptsize|/2$.
		'''
		ptsize = this.ptsize if type(this.ptsize)==int else float(this.ptsize[:-2])

		x = float(this.loc_x(this,i))*float(this.scale) + float(this.scale)*float(this.width)/2 + float(this.padding)
		y = 2*float(this.padding)+float(this.height)*float(this.scale)-(float(this.loc_y(this,i))*float(this.scale) + float(this.padding))

		this.canvas.create_oval(x-ptsize/2,y-ptsize/2,x+ptsize/2,y+ptsize/2, fill=this.color)
		return

	def validate(this):
		r'''
		Validates and corrects any variables on \verb|this| that may need preprocessing before drawing.
		'''
		if type(this.node_options)==str:
			node_options = this.node_options
			this.node_options = lambda hd,i: node_options
		if type(this.line_options)==str:
			line_options = this.line_options
			this.line_options = lambda hd,i: line_options

	@requires(tk)
	def tkinter(this, **kwargs):
		r'''
		Opens a window using tkinter and draws the Hasse diagram.

		The keyword arguments are described in \verb|HasseDiagram|.
		'''
		#save default parameters to restore aferwards
		defaults = this.__dict__.copy()
		#update parameters from kwargs
		this.__dict__.update(kwargs)
		this.validate()
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

		for i,J in this.P.covers(True).items():
			xi = float(this.loc_x(this,i))*this.scale + width/2 + this.padding
			yi = 2*this.padding+height-(float(this.loc_y(this,i))*this.scale + this.padding)
			for j in J:
				xj = float(this.loc_x(this,j))*this.scale + width/2 + this.padding
				yj = 2*this.padding+height-(float(this.loc_y(this,j))*this.scale + this.padding)
				canvas.create_line(xi,yi-this.scale*this.offset,xj,yj+this.scale*this.offset)#,color=this.color)
		root.mainloop() #makes this function blocking so you can actually see the poset when ran in a script
		this.__dict__.update(defaults)

	def latex(this, **kwargs):
		r'''
		Returns a string to depict the Hasse diagram in \LaTeX.

		The keyword arguments are described in |HasseDiagram|.
		'''
		defaults = this.__dict__.copy()
		this.__dict__.update(kwargs)
		this.validate()
		this.in_latex = True

		if len(this.P.ranks)==0:
			this.maxrksize = 0
		else:
			this.maxrksize = max([len(r) for r in this.P.ranks])

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

		if not this.labels:
			##############
			#write coords for elements
			##############
			for i in range(len(this.P)):
				ret.append('\\coordinate('+this.nodeName(this,i)+')at('+this.loc_x(this,i)+','+this.loc_y(this,i)+');\n')

			##############
			#draw lines for covers
			##############
			for i,J in this.P.covers(True).items():
				for j in J:
					options=this.line_options(this,i,j)
					if len(options)>0: options='['+options+']'
					ret.append('\\draw[color='+this.color+']'+options+'('+this.nodeName(this, i)+this.lowsuffix+')--('+this.nodeName(this, j)+this.highsuffix+");\n")
		###############
		#write nodes for the poset elements
		###############
			for rk in this.P.ranks:
				for r in rk:
					name=this.nodeName(this, r)
					ret.append('\\fill['+this.node_options(this,r)+']('+name+')circle('+this.ptsize+');\n')
#					ret.append('\\coordinate('+name+')at('+this.loc_x(this, r)+','+this.loc_y(this, r)+');\n')
#					ret.append('\\fill['+this.node_options(this,r)+']('+name+')circle('+this.ptsize+');\n')
		else: #this.labels==True
			for rk in this.P.ranks:
				for r in rk:
#					ret.append('\\node['+this.node_options(this,r)+']('+this.nodeName(this, r)+')at('+this.nodeName(this,r)+')\n{')
					ret.append('\\node['+this.node_options(this,r)+']('+this.nodeName(this, r)+')at('+this.loc_x(this, r)+','+this.loc_y(this, r)+')\n{')
					ret.append('\\scalebox{'+str(this.nodescale)+"}{")
					ret.append(str(r) if this.indices_for_nodes else this.nodeLabel(this, r))
					ret.append('}};\n\n')
			##############
			#draw lines for covers
			##############
			for i,J in this.P.covers(True).items():
				for j in J:
					options=this.line_options(this,i,j)
					if len(options)>0: options='['+options+']'
					ret.append('\\draw[color='+this.color+']'+options+'('+this.nodeName(this, i)+this.lowsuffix+')--('+this.nodeName(this, j)+this.highsuffix+");\n")
		##############
		##############
		ret.append('\\end{tikzpicture}')
		if this.standalone:
			ret.append('\n\\end{document}')

		this.__dict__.update(defaults)
		return ''.join(ret)
##############
#begin MinorPosetHasseDiagram class
##############

class MinorPosetHasseDiagram(HasseDiagram):
	r'''
	@is_section@no_children@
	TODO \verb|__doc__| string
	'''
	def __init__(this, P, L, **kwargs):
		super().__init__(P, **kwargs)
		this.P = P
		this.L = L
		this.L.hasseDiagram.__dict__.update({k[5:] : v for k,v in kwargs.items() if k[:5]=='latt_'})

	def latex(this, **kwargs):
		latt_args = {k[5:] : v for k,v in kwargs.items() if k[:5] == 'latt_'}
		latt_defaults = this.L.hasseDiagram.__dict__.copy()
		this.L.hasseDiagram.__dict__.update(latt_args)
		this.L.hasseDiagram.nodeName = MinorPosetHasseDiagram.latt_nodeName

		ret = super().latex(**kwargs)

		this.L.hasseDiagram.__dict__.update(latt_defaults)

		return ret

	def nodeLabel(this, i):
		if i in this.P.min(): return '$\\emptyset$'
		args = {
			'node_options' : MinorPosetHasseDiagram.make_node_options(this.P[i]),
			'line_options' : MinorPosetHasseDiagram.make_line_options(this.P[i]),
			}
		args.update({k[5:] : v for (k,v) in this.__dict__.items() if k[:5]=='latt_'})
		minorLatex = this.L.latex(**args)
		minorLatex = ''.join(minorLatex.split('\n')[2:-1])
		return '\\begin{tikzpicture}\\begin{scope}\n'+minorLatex+'\n\\end{scope}\\end{tikzpicture}'

	def latt_nodeName(this, i):
		return 'latt_'+HasseDiagram.nodeName(this,i)

	def make_node_options(KH):
		def node_options(this, i):
			if this.P.elements[i] in KH: return 'color=black'
			return 'color=gray'
		return node_options

	def make_line_options(KH):
		def line_options(this, i, j):
			if this.P.elements[i] in KH and this.P.elements[j] in KH: return 'color=black'
			return 'color=gray'
		return line_options


##############
#end MinorPosetHasseDiagram class
##############

