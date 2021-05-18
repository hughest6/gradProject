import math
pi = math.pi


# Square plate radar cross section
def plate_rcs(plate_area, theta, freq):
    gamma = 3E8/freq
    k = 2
    return (4*pi*(plate_area**4))/(gamma**2)*((math.sin(k*plate_area*math.sin(theta)))/(k*plate_area*math.sin(theta)))


# Cylinder cross section
def cylinder_rcs(cyl_area, cyl_length, freq):
    gamma = 3E8 / freq
    return (2 * pi * cyl_area * (cyl_length**2)) / (gamma**2)


# Trihedral corner reflector cross section
def trihedral_rcs(area, freq):
    gamma = 3E8 / freq
    return (4 * pi * (area ^ 4)) / (3 * (gamma**2))
