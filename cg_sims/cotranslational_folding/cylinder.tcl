# -------------------------------------------------------------------
# VMD Tcl script: Draw a cylinder guide
# Parameters
set cyl_radius 7.5
set cyl_length 100.0   ;# total length
set cyl_obj -1         ;# graphics object id (for deletion later)

# Procedure to draw the cylinder
proc cyl_draw {} {
    global cyl_radius cyl_length cyl_obj
    if { $cyl_obj != -1 } {
        graphics top delete $cyl_obj
    }
    # choose a transparent material
    graphics top material Transparent
    graphics top color blue
    set cyl_obj [graphics top cylinder \
        {0 0 0} \
        [list 0 0 $cyl_length] \
        radius $cyl_radius resolution 40 filled yes]
    puts "Cylinder drawn (radius=$cyl_radius, length=$cyl_length)"
}

# Procedure to hide the cylinder
proc cyl_hide {} {
    global cyl_obj
    if { $cyl_obj != -1 } {
        graphics top delete $cyl_obj
        set cyl_obj -1
        puts "Cylinder hidden"
    } else {
        puts "No cylinder to hide"
    }
}

# Procedure to toggle cylinder
proc cyl_toggle {} {
    global cyl_obj
    if { $cyl_obj == -1 } {
        cyl_draw
    } else {
        cyl_hide
    }
}
# -------------------------------------------------------------------
 
cyl_draw