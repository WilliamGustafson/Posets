'''@no_doc@no_children@'''
import random
import math
from .utils import *
from . import poset
import tkinter as tk

class HasseDiagram:
	r'''
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

	There are two types of options: constant values such as
	\verb|height|,\verb|width| or \verb|scale| and function
	values such as \verb|loc_x|, \verb|loc_y| or \verb|nodeLabel|.

	\textbf{\large Keyword arguments}

	Options that affect both \verb|latex()| and \verb|tkinter()|:

		\begin{itemize}

		\item[]{\verb|width| -- Width of the diagram. When calling \verb|latex()| this is the width
			in tikz units (by default centimeters), for \verb|tkinter()| the units are $\frac{1}{30}$th of tkinter's units.

			The default value is 8.
		}

		\item[]{\verb|height| -- Height of the diagram, uses the same units as width.

			The default value is 10.
		}

		\item[]{\verb|labels| -- If this is \verb|True| display labels, obtained from \verb|nodeLabels|, for elements; if this is \verb|False| display filled circles for elements.
			The default value is \verb|True|.
		}

		\item[]{\verb|ptsize| -- No effect when \verb|labels| is \verb|True|, when \verb|labels| is \verb|False| this is the size
			of the circles shown for elements. When calling \verb|tkinter()| this can
			be either a number or a string. For compatibility if
			\verb|ptsize| is a string the last two characters
			are ignored. When calling \verb|latex()| this should be a string and
			include units.

			The default value is '2pt'.
		}

		\item[]{\verb|indices_for_nodes| -- If \verb|True| then
			\verb|this.nodeLabel| is not called and the
			node text is the index of the element in the poset.
			If \verb|labels| is \verb|False| this argument has no effect.

			The default value is \verb|False|.
		}

		\item[]{\verb|nodeLabel| -- A function that given \verb|this| and an index returns a string to label
			the corresponding element by.

			The default value is \verb|HasseDiagram.nodeLabel|.
		}

		\item[]{\verb|loc_x| -- A function that given \verb|this| and an index returns the $x$-coordinate
			of the element in the diagram as a string. Positive values extend rightward and negative leftward.

			The default value is \verb|HasseDiagram.loc_x|.
		}

		\item[]{\verb|loc_y| -- A function that given \verb|this| and an index returns the $y$-coordinate
			of the element in the diagram as a string. Positive values extend upward and negative values downward.

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
			to the end of your implementation of \verb|loc_x| such as
				\begin{center}
				\verb|x = x+random.uniform(-this.jiggle-this.jiggle_x,this.jiggle+this.jiggle_x)|
				\end{center}

			The default values are 0.
		}

		\item[]{\verb|scale| -- In \verb|latex()|
			this is the scale parameter for the tikz environment, i.e. the
			tikz environment containing the figure begins
			\begin{center}
				\verb|'\\begin{tikzpicture}[scale='+tikzscale+']'|
			\end{center}
			In \verb|tkinter()| all coordinates are scaled by $\verb|scale|$.

			The default value is '1', this parameter may be a string or a numeric type.
		}

		\end{itemize}

		Options that affect only \verb|latex()|:

	\begin{itemize}
		\item[]{\verb|preamble| -- A string that when calling \verb|latex()| is placed in the preamble.
			It should be used to include any extra packages or define commands
			needed to produce node labels. This has no effect when standalone is \verb|False|.

			The default value is \verb|''|.
		}

		\item[]{\verb|nodescale| -- Each node is wrapped in \verb|'\\scalebox{'+nodescale+'}'|.

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
			is \verb|True| this has no effect and \verb|'.south'| is used for the low suffix.

			The default is \verb|''|.
		}

		\item[]{\verb|highsuffix| -- This is the suffix for the bottom node in each cover. If \verb|northsouth|
			is \verb|True| this has no effect and \verb|'.north'| is used for the high suffix.

			The default is \verb|''|.
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

			When \verb|False| only the code for the figure is returned, which case the return value
			begins with \verb|\begin{tikzpicture}| and ends with \verb|\end{tikzpicture}|.

			The default is \verb|False|.
		}
	\end{itemize}

	Options that affect only \verb|tkinter()|:
	\begin{itemize}
		\item[]{\verb|padding| -- A border of this width is added around all sides of the diagram.
			This is affected by \verb|scale|.

			The default value is 3.
		}

		\item[]{\verb|offset| -- Cover lines start above the bottom element and end below the top
			element, this controls the separation.

			The default value is 0.5.
		}

		\item[]{\verb|nodeDraw| -- When labels is \verb|False| this function is called instead of placing
			anything for the node. The function is passed \verb|this| and an index to
			the element to be drawn. \verb|nodeDraw| should use the \verb|tkinter.Canvas|
			object \verb|this.canvas| to draw. The center of your diagram should be
			at the point with coordinates given below.
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

	When overriding function parameters the first argument is always the \verb|HasseDiagram|
	instance. The class \verb|HasseDiagram| has an attribute for each of the options described above as well
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
	so if you intend to call these functions you must pass \verb|this| as an argument via
		\[\verb|this.nodeLabel(this, i)|\]
	The class methods remain unchanged of course, for example \verb|HasseDiagram.nodeLabel|
	always refers to the default implementation.

	\textbf{\large Subclassing}

	The second way to override a function parameter is via subclassing. This is more
	convenient if overriding several function parameters at once or if the computations
	are more involved. It is also useful for adding extra parameters. Any variables initialized
	in the constructor are saved at the beginning of \verb|latex()| or \verb|tkinter()|, overriden during
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
	@is_section@
	'''
	def __init__(this, P=None, that=None,**kwargs):
		'''
		See HasseDiagram.
		'''
		if that!=None:
			for attr in dir(that):
				if attr[:2]!='__':
					setattr(this,attr,getattr(that,attr))
			if P!=None: this.P = P
		else:
			this.P = P
			this.in_latex = False
			this.in_tkinter = False

			this.defaults = {
				'preamble':'',
				'nodescale':'1',
				'scale':'1',
				'line_options':'',
				'northsouth':True,
				'lowsuffix':'',
				'highsuffix':'',
				'labels': True,
				'ptsize': '2pt',
				'height': 10,
				'width': 8,
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
				'padding': 1,
				'nodeDraw': type(this).nodeDraw,
				'nodeTikz': type(this).nodeTikz,
				'offset': 0.5,
				'color':'black',
				'landscape': False,
				}

		for (k,v) in this.defaults.items():
			if k in kwargs: this.__dict__[k] = kwargs[k]
			else: this.__dict__[k] = v

		this.validate()

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
		ratio of the logarithms of the size of the rank to the size of the largest rank.

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

		This evenly spaces ranks. The return value is a string.
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

		This draws a filled black circle of radius $\verb|ptsize/2|$.
		'''
		ptsize = this.ptsize if type(this.ptsize)==int else float(this.ptsize[:-2])

		x = float(this.loc_x(this,i))*float(this.scale) + float(this.scale)*float(this.width)/2 + float(this.padding)
		y = 2*float(this.padding)+float(this.height)*float(this.scale)-(float(this.loc_y(this,i))*float(this.scale) + float(this.padding))

		this.canvas.create_oval(x-ptsize/2,y-ptsize/2,x+ptsize/2,y+ptsize/2, fill=this.color)
		return

	def nodeTikz(this,i):
		r'''
		This is the default implementation of \verb|nodeTikz| used to draw nodes when \verb|labels| is \verb|False|.
		'''
		return '\\fill['+this.node_options(this,i)+']('+this.nodeName(this,i)+')circle('+this.ptsize+');\n'

	def validate(this):
		r'''
		Validates and corrects any variables on \verb|this| that may need preprocessing before drawing.
		'''
		if type(this.node_options)==str:
			node_options = this.node_options
			this.node_options = lambda hd,i: node_options
			this.node_options.__name__ = "'"+node_options+"'"
		if type(this.line_options)==str:
			line_options = this.line_options
			this.line_options = lambda hd,i,j: line_options
			this.line_options.__name__ = "'"+line_options+"'"

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

		if len(this.P.ranks)==0:
			this.maxrksize = 0
		else:
			this.maxrksize = max([len(r) for r in this.P.ranks])

		root = tk.Tk()
		root.title("Hasse diagram of "+(this.P.name if hasattr(this.P,"name") else "a poset"))
		this.scale = float(this.scale)*30
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

		The keyword arguments are described in \verb|HasseDiagram|.
		'''
		defaults = this.__dict__.copy()
		this.__dict__.update(kwargs)
		this.validate()
		this.in_latex = True

		#right now landscape option is bugged disable it
		#until we can be bothered to fix it
		this.landscape = False

		if len(this.P.ranks)==0:
			this.maxrksize = 0
		else:
			this.maxrksize = max([len(r) for r in this.P.ranks])

		if this.northsouth and 'lowsuffix' not in kwargs and 'highsuffix' not in kwargs:
			this.lowsuffix = '.east' if this.landscape else '.north'
			this.highsuffix = '.west' if this.landscape else '.south'
		if this.landscape:
			temp = this.loc_x
			this.loc_x = this.loc_y
			temp = this.loc_x
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
			ret.append(this.preamble)
			ret.append('\n\\usepackage[psfixbb,graphics,tightpage,active]{preview}\n')
			ret.append('\\PreviewEnvironment{tikzpicture}\n\\usepackage[margin=0in]{geometry}\n')
			ret.append('\\begin{document}\n\\pagestyle{empty}\n')
		ret.append('\\begin{{tikzpicture}}[scale={}]\n'.format(this.scale))

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
					ret.append('\\draw'+options+'('+this.nodeName(this, i)+')--('+this.nodeName(this, j)+");\n")
		###############
		#write nodes for the poset elements
		###############
			for rk in this.P.ranks:
				for r in rk:
					#name=this.nodeName(this, r)
					ret.append(this.nodeTikz(this,r))
					#ret.append('\\fill['+this.node_options(this,r)+']('+name+')circle('+this.ptsize+');\n')
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
					ret.append('\\draw'+options+'('+this.nodeName(this, i)+this.lowsuffix+')--('+this.nodeName(this, j)+this.highsuffix+");\n")
		##############
		##############
		ret.append('\\end{tikzpicture}')
		if this.standalone:
			ret.append('\n\\end{document}')

		this.__dict__.update(defaults)
		return ''.join(ret)
##############
#begin SubposetsHasseDiagram class
##############

class SubposetsHasseDiagram(HasseDiagram):
	r'''
	This is a class used to draw posets whose elements are themselves subposets of some global poset, such as interval posets or lattices of ideals.

	The elements of the poset $P$ to be drawn are subposets
	of a poset $Q$.
	The nodes in the Hasse diagram of $P$ are represented as
	posets. The entire poset $Q$ is drawn for each element of $P$,
	the elements and
	edges contained in the given subposet are drawn in black and
	elements and edges not contained in the subposet are drawn
	in gray.

	Options can be passed to this class in order to control the
	drawing of the diagram in the same way as for the class
	\verb|HasseDiagram|. For example, calling \verb|latex(width=5)|
	on an instance of \verb|SubposetsHasseDiagram| sets the
	width of the entire diagram (that of $P$) to 5. To control
	options for the subposets a prefix, by default \verb|'Q'|,
	is used. For example, \verb|latex(Q_width=5,width=40)| would set
	the width of each subposet to 5 and the width of the
	entire diagram to 40.

	@is_section@subclass@
	'''
	def __init__(this, P, Q, is_in=lambda x,X:x in X, prefix='Q', draw_min=True, func_args=None, **kwargs):
		r'''
		Constructor arguments:
		\begin{itemize}
			\item[]{\verb|prefix| -- String to prefix options
				to be passed to the instances of
				\verb|HasseDiagram| that draw the subdiagrams.

				The argument \verb|prefix| should be
				a valid tikz node name. It is recommended
				that \verb|prefix| is also a valid python
				variable name.
				}
			\item[]{\verb|is_in| -- A function used by the
			constructor to test whether elements of \verb|Q| are
			elements of a subposet. The function
			\verb|is_in| takes two arguments, an element \verb|x| of the poset $Q$ and the subposet object \verb|X| to test
			containment with. The default value returns
			\verb|x in X|.}
			\item[]{\verb|draw_min| -- If \verb|True| all elements
				of \verb|P| are represented by a Hasse diagram.
				If \verb|False| minimal elements are not
				drawn but instead labeled by the return
				value of \verb|this.minNodeLabel|.
				}
			\item[]{\verb|func_args| -- A dictionary whose keys are names of keyword arguments
				to \verb|HasseDiagram.latex| and whose values are functions that take
				this instance of \verb|SubposetsHasseDiagram| and an index into the
				poset \verb|P|. When the subposet for an element \verb|p| at index
				\verb|i| in \verb|P| is drawn both \verb|this| and \verb|i| are passed
				to each function and the corresponding option is set to the return
				value when drawing the subposet.
				}
		\end{itemize}

		All keyword arguments not beginning with the string
		\verb|this.prefix+'_'|
		are handled the same as in the
		class \verb|HasseDiagram|. Keyword arguments that
		begin with the string \verb|this.prefix+'_'|
		are saved as attributes and passed to the instances
		of \verb|HasseDiagram| drawing the subposets
		when \verb|latex()| is called.
		'''
		this.prefix=prefix+'_'
		this.prefix_len = len(this.prefix)
		this.minNodeLabel = type(this).minNodeLabel
		super().__init__(P,**kwargs)
		this.P = P
		this.Q = Q
		this.is_in = is_in
		this.Q.hasseDiagram.__dict__.update({k[len(this.prefix):] : v for k,v in kwargs.items() if k[:len(this.prefix)]==this.prefix})
		this.func_args = {} if func_args is None else func_args
		this.draw_min = draw_min

	def latex(this, **kwargs):
		r'''
		Returns latex code for the poset attached to this object.

		See \verb|HasseDiagram.latex| and \verb|SubposetsHasseDiagram|.
		'''
		Q_args = {k[len(this.prefix):] : v for k,v in kwargs.items() if k[:len(this.prefix)]==this.prefix}
		Q_defaults = this.Q.hasseDiagram.__dict__.copy()
		this.Q.hasseDiagram.__dict__.update(Q_args)
		this.Q.hasseDiagram.nodeName = SubposetsHasseDiagram.Q_nodeName
		this.Q.hasseDiagram.prefix = this.prefix

		ret = super().latex(**kwargs)

		this.Q.hasseDiagram.__dict__.update(Q_defaults)

		return ret

	def minNodeLabel(this):
		r'''
		Returns \verb|r'$\emptyset$'|.

		This function is called by \verb|nodeLabel| to get a node label for minimal elements if \verb|draw_min| is \verb|False|.
		To change the label for minimal elements provide your own version of \verb|minNodeLabel|.
		'''
		return r'$\emptyset$'

	def nodeLabel(this, i):
		r'''
		Default implementation of \verb|nodeLabel| for \verb|SubposetsHasseDiagram|.

		This returns a list of elements if \verb|this.in_tkinter|
		is \verb|True| and otherwise returns
		tikz code for the poset \verb|this.P[i]|.
		'''
		if not this.in_latex:
			return ','.join(str(x) for x in this.P[i])
		if not this.draw_min and i in this.P.min(): return this.minNodeLabel(this)
		args = {
			'node_options' : SubposetsHasseDiagram.make_node_options(this.P[i]),
			'line_options' : SubposetsHasseDiagram.make_line_options(this.P[i]),
			}
		args.update({k[len(this.prefix):] : v for (k,v) in this.__dict__.items() if k[:len(this.prefix)]==this.prefix})
		func_arg_values = {k:v(this,i) for k,v in this.func_args.items()}
		args.update(func_arg_values)
		
		args['parent']=this
		args[this.prefix[:-1]] = this.P[i]
		Q_Latex = this.Q.latex(**args)
		try:
			start = Q_Latex.index('\\begin{tikzpicture}')+len('\\begin{tikzpicture}')
			start += Q_Latex[start:].index(']') + 1
		except:
			start = 0
		Q_Latex = Q_Latex[start:]
		try:
			end = Q_Latex.index('\\end{tikzpicture}')
		except:
			end = -1
		Q_Latex = Q_Latex[:end]
		return '\\begin{{tikzpicture}}[scale={}]\\begin{{scope}}\n'.format(this.Q.hasseDiagram.scale)+Q_Latex+'\n\\end{scope}\\end{tikzpicture}'

	def Q_nodeName(this, i):
		r'''
		Returns a node name for an element of $P$.

		To ensure the node names of the larger figure
		and of the subdiagrams do not clash
		all node names are prefixed with \verb|this.prefix|.
		'''
		return this.prefix+HasseDiagram.nodeName(this,i)

	def make_node_options(q):
		r'''
		Returns a function to be supplied as \verb|node_options| in the \verb|latex()| call to draw a diagram for \verb|q|.
		'''
		def node_options(this, i):
			if this.parent.is_in(this.P.elements[i],q): return 'color=black'
			return 'color=gray'
		return node_options

	def make_line_options(q):
		r'''
		Returns a function to be supplied as \verb|line_options| in the \verb|latex()| call to draw a diagram for \verb|q|.
		'''
		def line_options(this, i, j):
			if this.parent.is_in(this.P.elements[i],q) and this.parent.is_in(this.P.elements[j],q): return 'color=black'
			return 'color=gray'
		return line_options
##############
#end SubposetsHasseDiagram class
##############
class ZetaHasseDiagram(SubposetsHasseDiagram):
	r'''
	Class to draw the Hasse diagram of a poset as principal filters (or ideals) labeled by the zeta function values.

	This is a convenience class that merely passes appropriate options to \verb|SubposetsHasseDiagram|.
	When \verb|latex| is called this class produces latex code for the Hasse diagram of the given
	poset $P$ with each element $p$ as the principal filter $\{q\in P:q\ge p\}$ (or optionally the principal
	ideal) with each element $q$ in the filter labeled by $\zeta(p,q)$.

	This class is intended for representing quasigraded posets, those with a zeta function taking values
	other than 0 and 1.

	Constructor arguments:
	\begin{itemize}
		\item[]{\verb|filters| -- Whether to represent elements by the associated principal
			filter or alternatively as ideals. The default value is \verb|True| which
			will use filters.
			}
		\item[]{\verb|keep_ranks| -- Whether to use the same rank values for elements in
			the filters/ideals drawn as in the given poset. If this argument is \verb|False|
			then a new poset is created with rank function the standard length function
			as returned by \verb|Poset.make_ranks|.
			}
	\end{itemize}


	See \verb|SubposetsHasseDiagram| for details on other arguments. Note, the argument \verb|prefix| to
	\verb|SubposetsHasseDiagram| defaults to \verb|'V'|.

	Note, if \verb|V_width| (or \verb|V_height|) is not provided (assuming
	the default value \verb|'V'| for \verb|prefix|) it is set to one fifth
	of \verb|width| (or \verb|height|).
	If \verb|V_nodescale| is not provided it is set to \verb|0.5|.

	@is_section@subclass@
	'''
	def __init__(this, P, filters=True, prefix='V', keep_ranks=True, func_args=None,\
	**kwargs):
		r'''
		See \verb|ZetaHasseDiagram|.
		'''
		if func_args is None:
			if filters:
				func_args = {'nodeLabel' : lambda Hd,i:lambda hd,j:'0' if i>j else str(hd.P.zeta[i,j])}
			else:
				func_args = {'nodeLabel' : lambda Hd,i:lambda hd,j:'0' if j>i else str(hd.P.zeta[j,i])}
		#option for ideals instead of filters but use filters for terminology below
		if keep_ranks: Q=poset.Poset(zeta=P.zeta,elements=P.elements,name=P.name,ranks=P.ranks)
		else: Q=poset.Poset(zeta=P.zeta,elements=P.elements)
		super().__init__(P=P,Q=Q, prefix=prefix, func_args=func_args, is_in=(lambda x,y:P.lesseq(y,x)) if filters else P.lesseq)
		if f'{this.prefix}height' not in kwargs:
			this.__dict__[f'{this.prefix}height'] = float(this.height) / 5
		if f'{this.prefix}width' not in kwargs:
			this.__dict__[f'{this.prefix}width'] = float(this.width) / 5
		if f'{this.prefix}nodescale' not in kwargs:
			this.__dict__[f'{this.prefix}nodescale'] = 0.5
