# Magnetometry Neural Networks
It's often beneficial for research, engineering, or other teams to be able to calculate a magnetic field at any arbitrary point in space. One of the most common ways of calculating complex fields is by use of some 3D software like COMSOL or, in my case, Opera3D. In these softwares, you can build a full 3D construction of coils and materials and have it calculate the field anywhere.

The biggest problem with using 3D softwares is that, a lot of the time, these magnetic fields might need to be used in an analysis campaign. That becomes an issue because most of these 3D simulations don't have an easy way to talk to your code where you can have your code ask for the field at a specific point, have the software return the field, and repeat. If we can instead use these programs to train a machine learning model that can map the magnetic field itself, the computations can not only be very efficient, but it can hopefully generalize the entire field in code-form and allow easy integration into analysis.

Sometimes these 3D simulations are not available to a team, either due to pricing or, most often, complexity of the setup. In these cases, individuals will need to take magnetic field measurements to try to understand their field profile. If a neural network can generalize these data points and build an entire field from these data points, it could serve as an invaluable tool in the magnetometry world.

## Uniqueness Theorem in Magnetism
I'm working on a document that will cover the details of this section a little bit more in depth, but for now we should cover a key component of this effort called the uniqueness theorem. Let's begin with Maxwell's equations, the equations that describe the behavior of magnetic fields. If we select a region that has no currents or changing electric fields within the interior, then Maxwell's equations simplify to:

$\nabla \cdot \textbf{B} = 0$
 
$\nabla \times \textbf{B} = 0$

Again, details will be saved for another document, but these equations basically mean that, within this region, there is a unique solution to Maxwell's equations. The importance of this is crucial. That means that, if we can get a NN to accurately map the field on the boundary of this region accurately, then the entire field that it predicts inside must also be accurate. If it produced a solution that worked on the boundary, that one solution **must** be equal to the unique solution to Maxwell's equations.

Due to this theorem, we train our model on points on a boundary, which allows for a higher density of points since we are working in 2 dimensions instead of 3. A higher density of points should mean a more accurate model.

## Regions
As this project is being done for the Nab experiment at Oak Ridge National Laboratory in Oak Ridge, Tennessee, we are trying to map the field at this experiment. This means we fall under restrictions imposed by the physical setup of the experiment. Let's talk about what this means without delving too deep into what the Nab experiment is about (I'd like to add this extra information to the same document that would cover the uniqueness theorem).

We have 16 concentric coils producing this field. They are designed in such a way that our field lines have an outer radius that caps out at about 10cm. Centered at z=0cm, however, is a smaller coil called the filter coils (F coil). This coil has an inner radius of right around 4cm. 

What does this mean for our NN project? Well in an ideal world, we'd just train the model on the boundary of a cylinder of 10cm radius since that's the farthest out field lines. But doing this would mean that our boundary contains the F coil inside of it. This, in turn means that current exists within our region which means that Maxwell's equations as they're written above do not hold. The uniqueness theorem would also no longer hold under such conditions.

It is for this reason that we split our region-of-interest into 3 sections. The UDET (Upper Detector) region will range from 3cm <= z <= 550cm and 0cm <= r <= 10cm. The Filter region, made smaller to stay within the F coil, will range from -3cm <= z <= 3cm and 0cm <= r <= 3cm. Once we've made it far enough away from the F coil, we expand outwards again for the LDET (Lower Detector) region ranging from -150cm <= z <= -3cm and 0cm <= r <= 10cm.

The splitting of 3 regions means the separate training of 3 models. When we need the field, we'll just decide which model to ask the field from based on the coordinates in question.

## Cylindrical vs Rectangular
I did mention cylindrical symmetry and cylindrical surfaces but I've been a bit misleading. While our coils have cylindrical symmetry in their fields, our magnetic shielding is **not** cylindrically symmetric. Partially for that reason, but also since cartesian is just easier to work with in Opera3D as well as calculating the divergence and curl computationally, I choose to use cartesian coordinates.

What does this mean for training? Well it just means that we're training on the boundary of a rectangular box rather than a cylinder. It'll be a cylindrical-ish box though! The x and y widths will always be the same just like the radius of a cylinder is! So when you see me use cartesian coordinates, there's the reason.