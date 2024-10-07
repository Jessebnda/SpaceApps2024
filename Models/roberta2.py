from transformers import pipeline

# Configurar el pipeline de preguntas y respuestas
pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Contexto que se utilizará para responder preguntas
context = """
{
        "name": "Sun",
        "type": "star",
        "position": 0,
        "Vol. Mean Radius (km)": "695700",
        "Density (g cm^-3)": "1.408",
        "Mass x10^30 (kg)": "1.989",
        "Volume (x10^10 km^3)": "1.412",
        "Sidereal rot. period (d)": "25.38",
        "Sid. rot. rate (rad/s)": "2.865x10^-6",
        "Mean solar day (d)": "24.47",
        "Core radius (km)": "~150000",
        "Geometric Albedo": "N/A",
        "Surface emissivity": "N/A",
        "GM (km^3/s^2)": "132712440018",
        "Equatorial radius, Re (km)": "695700",
        "GM 1-sigma (km^3/s^2)": "N/A",
        "Mass ratio (Sun/planet)": "N/A",
        "Moment of Inertia": "N/A",
        "Equatorial gravity (m/s^2)": "274.0",
        "Atmospheric pressure (bar)": "N/A",
        "Max. angular diameter (arcsec)": "32.0",
        "Mean Temperature (K)": "5778",
        "Visual mag. V(1,0)": "-26.74",
        "Obliquity to orbit": "7.25 degrees",
        "Hill's sphere radius (Rp)": "N/A",
        "Sidereal orbital period (y)": "N/A",
        "Mean orbital velocity (km/s)": "N/A",
        "Sidereal orbital period (d)": "N/A",
        "Escape velocity (km/s)": "617.7",
        "Solar Constant (W/m^2)": {
            "Perihelion": "1367",
            "Aphelion": "1322",
            "Mean": "1350"
        },
        "Maximum Planetary IR (W/m^2)": {
            "Perihelion": "N/A",
            "Aphelion": "N/A",
            "Mean": "N/A"
        },
        "Minimum Planetary IR (W/m^2)": {
            "Perihelion": "N/A",
            "Aphelion": "N/A",
            "Mean": "N/A"
        }
    },
    {
        "name": "Mercury",
        "type": "planet",
        "position": 1,
        "Vol. Mean Radius (km)": "2439.4+-0.1",
        "Density (g cm^-3)": "5.427",
        "Mass x10^23 (kg)": "3.302",
        "Volume (x10^10 km^3)": "6.085",
        "Sidereal rot. period (d)": "58.6463",
        "Sid. rot. rate (rad/s)": "0.00000124001",
        "Mean solar day (d)": "175.9421",
        "Core radius (km)": "~1600",
        "Geometric Albedo": "0.106",
        "Surface emissivity": "0.77+-0.06",
        "GM (km^3/s^2)": "22031.86855",
        "Equatorial radius, Re (km)": "2440.53",
        "GM 1-sigma (km^3/s^2)": " ",
        "Mass ratio (Sun/planet)": "6023682",
        "Moment of Inertia": "0.33",
        "Equatorial gravity (m/s^2)": "3.701",
        "Atmospheric pressure (bar)": "<5x10^-15",
        "Max. angular diameter (arcsec)": "11.0",
        "Mean Temperature (K)": "440",
        "Visual mag. V(1,0)": "-0.42",
        "Obliquity to orbit": "2.11' +/- 0.1'",
        "Hill's sphere radius (Rp)": "94.4",
        "Sidereal orbital period (y)": "0.2408467",
        "Mean orbital velocity (km/s)": "47.362",
        "Sidereal orbital period (d)": "87.969257",
        "Escape velocity (km/s)": "4.435",
        "Solar Constant (W/m^2)": {
            "Perihelion": "14462",
            "Aphelion": "6278",
            "Mean": "9126"
        },
        "Maximum Planetary IR (W/m^2)": {
            "Perihelion": "12700",
            "Aphelion": "5500",
            "Mean": "8000"
        },
        "Minimum Planetary IR (W/m^2)": {
            "Perihelion": "6",
            "Aphelion": "6",
            "Mean": "6"
        }
    },
    {
        "name": "Venus",
        "type": "planet",
        "position": 2,
        "Vol. Mean Radius (km)": "6051.8+-0.1",
        "Density (g cm^-3)": "5.204",
        "Mass x10^23 (kg)": "48.685",
        "Volume (x10^10 km^3)": "92.843",
        "Sidereal rot. period (d)": "243.025",
        "Sid. rot. rate (rad/s)": "-0.00029924",
        "Mean solar day (d)": "116.75",
        "Core radius (km)": "~3000",
        "Geometric Albedo": "0.65",
        "Surface emissivity": "0.95+-0.05",
        "GM (km^3/s^2)": "324858.599",
        "Equatorial radius, Re (km)": "6051.8",
        "GM 1-sigma (km^3/s^2)": " ",
        "Mass ratio (Sun/planet)": "408523.71",
        "Moment of Inertia": "0.33",
        "Equatorial gravity (m/s^2)": "8.87",
        "Atmospheric pressure (bar)": "92.1",
        "Max. angular diameter (arcsec)": "66",
        "Mean Temperature (K)": "737",
        "Visual mag. V(1,0)": "-4.47",
        "Obliquity to orbit": "177.4' +/- 0.1'",
        "Hill's sphere radius (Rp)": "162",
        "Sidereal orbital period (y)": "0.61519726",
        "Mean orbital velocity (km/s)": "35.02",
        "Sidereal orbital period (d)": "224.701",
        "Escape velocity (km/s)": "10.36",
        "Solar Constant (W/m^2)": {
            "Perihelion": "2614",
            "Aphelion": "2608",
            "Mean": "2611"
        },
        "Maximum Planetary IR (W/m^2)": {
            "Perihelion": "153",
            "Aphelion": "153",
            "Mean": "153"
        },
        "Minimum Planetary IR (W/m^2)": {
            "Perihelion": "153",
            "Aphelion": "153",
            "Mean": "153"
        }
    },
    {
        "name": "Mars",
        "type": "planet",
        "position": 4,
        "Vol. Mean Radius (km)": "3389.92+-0.04",
        "Density (g cm^-3)": "3.933(5+-4)",
        "Mass x10^23 (kg)": "6.4171",
        "Volume (x10^10 km^3)": "16.318",
        "Sidereal rot. period (d)": "24.622962 hr",
        "Sid. rot. rate (rad/s)": "0.0000708822",
        "Mean solar day (d)": "88775.24415 s",
        "Core radius (km)": "~1700",
        "Geometric Albedo": "0.150",
        "GM (km^3/s^2)": "42828.375214",
        "Equatorial radius, Re (km)": "3396.19",
        "GM 1-sigma (km^3/s^2)": "+- 0.00028",
        "Mass ratio (Sun/planet)": "3098703.59",
        "Moment of Inertia": "",
        "Equatorial gravity (m/s^2)": "3.71",
        "Atmospheric pressure (bar)": "0.0056",
        "Max. angular diameter (arcsec)": "17.9",
        "Mean Temperature (K)": "210",
        "Visual mag. V(1,0)": "-1.52",
        "Obliquity to orbit": "25.19 deg",
        "Hill's sphere radius (Rp)": "319.8",
        "Sidereal orbital period (y)": "1.88081578",
        "Mean orbital velocity (km/s)": "24.13",
        "Sidereal orbital period (d)": "686.98",
        "Escape velocity (km/s)": "5.027",
        "Solar Constant (W/m^2)": {
            "Perihelion": "717",
            "Aphelion": "493",
            "Mean": "589"
        },
        "Maximum Planetary IR (W/m^2)": {
            "Perihelion": "470",
            "Aphelion": "315",
            "Mean": "390"
        },
        "Minimum Planetary IR (W/m^2)": {
            "Perihelion": "30",
            "Aphelion": "30",
            "Mean": "30"
        }
    },
    {
        "name": "Jupiter",
        "type": "planet",
        "position": 5,
        "Vol. Mean Radius (km)": "69911+-6",
        "Density (g/cm^3)": "1.3262 +- .0003",
        "Mass x10^22 (g)": "189818722 +- 8817",
        "Equatorial radius (1 bar)": "71492+-4 km",
        "Polar radius (km)": "66854+-10",
        "Flattening": "0.06487",
        "Geometric Albedo": "0.52",
        "Rocky core mass (Mc/M)": "0.0261",
        "Sidereal rot. period (III)": "9h 55m 29.711 s",
        "Sidereal rot. rate (rad/s)": "0.00017585",
        "Mean solar day, hrs": "~9.9259",
        "GM (km^3/s^2)": "126686531.900",
        "GM 1-sigma (km^3/s^2)": "+- 1.2732",
        "Equatorial gravity (m/s^2)": "24.79",
        "Polar gravity (m/s^2)": "28.34",
        "Visual mag. V(1,0)": "-9.40",
        "Visual mag. (opposition)": "-2.70",
        "Obliquity to orbit": "3.13 deg",
        "Sidereal orbital period (y)": "11.861982204",
        "Sidereal orbital period (d)": "4332.589",
        "Mean orbital speed (km/s)": "13.0697",
        "Atmospheric temperature (1 bar)": "165+-5 K",
        "Escape speed (km/s)": "59.5",
        "A_roche(ice)/Rp": "2.76",
        "Hill's sphere radius (Rp)": "740",
        "Solar Constant (W/m^2)": {
            "Perihelion": "56",
            "Aphelion": "46",
            "Mean": "51"
        },
        "Maximum Planetary IR (W/m^2)": {
            "Perihelion": "13.7",
            "Aphelion": "13.4",
            "Mean": "13.6"
        },
        "Minimum Planetary IR (W/m^2)": {
            "Perihelion": "13.7",
            "Aphelion": "13.4",
            "Mean": "13.6"
        }
    },
    {
        "name": "Saturn",
        "type": "planet",
        "position": 6,
        "Vol. Mean Radius (km)": "58232+-6",
        "Density (g cm^-3)": "0.687+-0.001",
        "Mass x10^26 (kg)": "5.6834",
        "Equat. radius (km)": "60268+-4",
        "Polar radius (km)": "54364+-10",
        "Sidereal rot. period (h)": "10h 39m 22.4s",
        "Sid. rot. rate (rad/s)": "0.000163785",
        "Mean solar day (hrs)": "10.656",
        "Core mass (Mc/M)": "0.1027",
        "Geometric Albedo": "0.47",
        "Surface emissivity": "",
        "GM (km^3/s^2)": "37931206.234",
        "Equatorial gravity (m/s^2)": "10.44",
        "Moment of Inertia": "",
        "Visual mag. V(1,0)": "-8.88",
        "Obliquity to orbit": "26.73°",
        "Hill's sphere radius (Rp)": "1100",
        "Sidereal orbital period (y)": "29.447498",
        "Mean orbital velocity (km/s)": "9.68",
        "Sidereal orbital period (d)": "10755.698",
        "Escape velocity (km/s)": "35.5",
        "Solar Constant (W/m^2)": {
            "Perihelion": "16.8",
            "Aphelion": "13.6",
            "Mean": "15.1"
        },
        "Maximum Planetary IR (W/m^2)": {
            "Perihelion": "4.7",
            "Aphelion": "4.5",
            "Mean": "4.6"
        },
        "Minimum Planetary IR (W/m^2)": {
            "Perihelion": "4.7",
            "Aphelion": "4.5",
            "Mean": "4.6"
        }
    },
    {
        "name": "Uranus",
        "type": "planet",
        "position": 7,
        "Vol. Mean Radius (km)": "25362+-12",
        "Density (g cm^-3)": "1.271",
        "Mass x10^24 (kg)": "86.813",
        "Equat. radius (km)": "25559+-4",
        "Polar radius (km)": "24973+-20",
        "Sidereal rot. period (h)": "17.24+-0.01",
        "Sid. rot. rate (rad/s)": "-0.000101237",
        "Mean solar day (h)": "~17.24",
        "Core mass (Mc/M)": "0.0012",
        "Geometric Albedo": "0.51",
        "GM (km^3/s^2)": "5793951.256",
        "Equatorial gravity (m/s^2)": "8.87",
        "Moment of Inertia": "",
        "Visual mag. V(1,0)": "-7.11",
        "Obliquity to orbit": "97.77 deg",
        "Hill's sphere radius (Rp)": "2700",
        "Sidereal orbital period (y)": "84.0120465",
        "Mean orbital velocity (km/s)": "6.8",
        "Sidereal orbital period (d)": "30685.4",
        "Escape velocity (km/s)": "21.3",
        "Solar Constant (W/m^2)": {
            "Perihelion": "4.09",
            "Aphelion": "3.39",
            "Mean": "3.71"
        },
        "Maximum Planetary IR (W/m^2)": {
            "Perihelion": "0.72",
            "Aphelion": "0.55",
            "Mean": "0.63"
        },
        "Minimum Planetary IR (W/m^2)": {
            "Perihelion": "0.72",
            "Aphelion": "0.55",
            "Mean": "0.63"
        }
    },
    {
        "name": "Neptune",
        "type": "planet",
        "position": 8,
        "Vol. Mean Radius (km)": "24624+-21",
        "Density (g cm^-3)": "1.638",
        "Mass x10^24 (kg)": "102.409",
        "Volume (x10^10 km^3)": "6254",
        "Sidereal rot. period (d)": "16.11+-0.01 hr",
        "Sid. rot. rate (rad/s)": "0.000108338",
        "Mean solar day (h)": "~16.11",
        "Core radius (km)": "24342+-30",
        "Geometric Albedo": "0.41",
        "Surface emissivity": "N/A",
        "GM (km^3/s^2)": "6835099.97",
        "Equatorial radius, Re (km)": "24766+-15",
        "GM 1-sigma (km^3/s^2)": "+-10",
        "Mass ratio (Sun/planet)": "N/A",
        "Moment of Inertia": "N/A",
        "Equatorial gravity (m/s^2)": "11.15",
        "Atmospheric pressure (bar)": "N/A",
        "Max. angular diameter (arcsec)": "N/A",
        "Mean Temperature (K)": "72+-2",
        "Visual mag. V(1,0)": "-6.87",
        "Obliquity to orbit": "28.32 deg",
        "Hill's sphere radius (Rp)": "4700",
        "Sidereal orbital period (y)": "164.788501027",
        "Mean orbital velocity (km/s)": "5.43",
        "Sidereal orbital period (d)": "60189",
        "Escape velocity (km/s)": "23.5",
        "Solar Constant (W/m^2)": {
            "Perihelion": "1.54",
            "Aphelion": "1.49",
            "Mean": "1.51"
        },
        "Maximum Planetary IR (W/m^2)": {
            "Perihelion": "0.52",
            "Aphelion": "0.52",
            "Mean": "0.52"
        },
        "Minimum Planetary IR (W/m^2)": {
            "Perihelion": "0.52",
            "Aphelion": "0.52",
            "Mean": "0.52"
        }
    },
    {
        "name": "Pluto",
        "type": "dwarf planet",
        "position": 9,
        "Vol. Mean Radius (km)": "1188.3+-1.6",
        "Density (g cm^-3)": "1.86",
        "Mass x10^22 (kg)": "1.307+-0.018",
        "Volume (x10^10 km^3)": "0.697",
        "Sidereal rot. period (h)": "153.29335198",
        "Sid. rot. rate (rad/s)": "0.0000113856",
        "Mean solar day (h)": "153.2820",
        "Core radius (km)": "N/A",
        "Geometric Albedo": "N/A",
        "Surface emissivity": "N/A",
        "GM (km^3/s^2)": "869.326",
        "Equatorial radius, Re (km)": "1188.3",
        "GM 1-sigma (km^3/s^2)": "0.4",
        "Mass ratio (Sun/planet)": "N/A",
        "Moment of Inertia": "N/A",
        "Equatorial gravity (m/s^2)": "0.611",
        "Atmospheric pressure (bar)": "N/A",
        "Max. angular diameter (arcsec)": "N/A",
        "Mean Temperature (K)": "N/A",
        "Visual mag. V(1,0)": "N/A",
        "Obliquity to orbit": "N/A",
        "Hill's sphere radius (Rp)": "N/A",
        "Sidereal orbital period (y)": "249.58932",
        "Mean orbital velocity (km/s)": "4.67",
        "Sidereal orbital period (d)": "N/A",
        "Escape velocity (km/s)": "1.21",
        "Solar Constant (W/m^2)": {
            "Perihelion": "1.56",
            "Aphelion": "0.56",
            "Mean": "0.88"
        },
        "Maximum Planetary IR (W/m^2)": {
            "Perihelion": "0.8",
            "Aphelion": "0.3",
            "Mean": "0.5"
        },
        "Minimum Planetary IR (W/m^2)": {
            "Perihelion": "0.8",
            "Aphelion": "0.3",
            "Mean": "0.5"
        }
    },
    {
    "name": "2688 Halley (1982 HG1)",
    "Rec #": "2688 (+COV)",
    "solution_date": "2024-Aug-01_12:49:17",
    "observation_count": "4847 (1950-2024)",
    "epoch": "2457983.5 (2017-Aug-18.00 TDB)",
    "orbital_elements": {
      "residual_rms": "0.24866",
      "eccentricity": "0.143291496006993",
      "perihelion_distance": "2.716225451628648",
      "time_of_perihelion_passage": "2457665.1965688192",
      "longitude_of_ascending_node": "95.24692196095351",
      "argument_of_perihelion": "188.315848445655",
      "inclination": "3.449232521284169",
      "semi_major_axis": "3.170536348091181",
      "mean_anomaly": "55.57075399091065",
      "aphelion_distance": "3.624847244553715",
      "orbital_period": "5.64556",
      "mean_motion": "0.174584213",
      "angular_momentum": "0.030313971",
      "ascending_node_distance": "3.61848",
      "descending_node_distance": "2.71981",
      "longitude": "283.5479163",
      "latitude": "-0.4985672",
      "minimum_orbit_intersection_distance": "1.69964004",
      "last_perihelion_passage": "2016-Oct-03.6965688192"
    },
    "physical_parameters": {
      "GM": "n.a.",
      "radius": "10.7625",
      "rotational_period": "16.022",
      "absolute_magnitude": "11.87",
      "phase_slope": "0.150",
      "albedo": "0.077",
      "spectral_type": "n.a."
    }
  },
  {
    "name": "Hale-Bopp (C/1995 O1)",
    "Rec #": "90002213 (+COV)",
    "solution_date": "2022-Aug-01_01:56:46",
    "observation_count": "66 (1993-2022)",
    "epoch": "2459837.5 (2022-Sep-15.0000000 TDB)",
    "orbital_elements": {
      "residual_rms": "n.a.",
      "eccentricity": "0.9949810027633206",
      "perihelion_distance": "0.890537663547794",
      "time_of_perihelion_passage": "2450537.1349071441",
      "longitude_of_ascending_node": "282.7334213961641",
      "argument_of_perihelion": "130.4146670659176",
      "inclination": "89.28759424740302",
      "semi_major_axis": "177.4333839117583",
      "mean_anomaly": "3.878386339423163",
      "aphelion_distance": "353.9762301599687",
      "orbital_period": "2363.5304681429",
      "mean_motion": "0.000417014",
      "angular_momentum": "0.02292857",
      "ascending_node_distance": "5.00538",
      "descending_node_distance": "1.07996",
      "longitude": "101.8968625",
      "latitude": "49.580132",
      "minimum_orbit_intersection_distance": "0.0878151",
      "last_perihelion_passage": "1997-Mar-29.6349071441"
    },
    "physical_parameters": {
      "GM": "n.a.",
      "radius": "30",
      "M1": "4.8",
      "M2": "n.a.",
      "k1": "4",
      "k2": "n.a.",
      "PHCOF": "n.a."
    }
  },
  {
    "name": "Hyakutake (C/1996 B2)",
    "Rec #": "90002219 (+COV)",
    "solution_date": "2021-Apr-15_23:29:47",
    "observation_count": "977 (306 days)",
    "epoch": "2450157.5 (1996-Mar-15.0000000 TDB)",
    "orbital_elements": {
      "residual_rms": "n.a.",
      "eccentricity": "0.9998916470450124",
      "perihelion_distance": "0.2302235310262354",
      "time_of_perihelion_passage": "2450204.8941449965",
      "longitude_of_ascending_node": "188.045131992156",
      "argument_of_perihelion": "130.1751209780967",
      "inclination": "124.9220493922234",
      "semi_major_axis": "2124.755444396066",
      "mean_anomaly": "359.9995230582502",
      "aphelion_distance": "4249.280665261106",
      "orbital_period": "97942.599927659",
      "mean_motion": "0.000010063",
      "angular_momentum": "0.011672383",
      "ascending_node_distance": "1.29717",
      "descending_node_distance": "0.27988",
      "longitude": "42.1829369",
      "latitude": "38.7916531",
      "minimum_orbit_intersection_distance": "0.100967",
      "last_perihelion_passage": "1996-May-01.3941449965"
    },
    "physical_parameters": {
      "GM": "n.a.",
      "radius": "2.1",
      "M1": "7.4",
      "M2": "11.1",
      "k1": "10.75",
      "k2": "5",
      "PHCOF": "0.030"
    },
    "non_gravitational_force_model": {
      "AMRAT": "0",
      "A1": "2.68769288063E-8",
      "A2": "6.515068560839E-10",
      "A3": "0",
      "DT": "0",
      "ALN": "0.1112620426",
      "NK": "4.6142",
      "NM": "2.15",
      "NN": "5.093",
      "R0": "2.808"
    }
  },
  {
    "name": "NEOWISE (C/2020 F3)",
    "Rec #": "90004571 (+COV)",
    "solution_date": "2024-Jul-30_14:57:24",
    "observation_count": "1315 (2020-2021)",
    "epoch": "2459036.5 (2020-Jul-06.0000000 TDB)",
    "orbital_elements": {
      "residual_rms": "n.a.",
      "eccentricity": "0.9991780262531292",
      "perihelion_distance": "0.2946512493809196",
      "time_of_perihelion_passage": "2459034.1788980444",
      "longitude_of_ascending_node": "61.01042818536988",
      "argument_of_perihelion": "37.2786584481257",
      "inclination": "128.9375027594809",
      "semi_major_axis": "358.4679565529321",
      "mean_anomaly": "0.0003370720801209784",
      "aphelion_distance": "716.6412618564833",
      "orbital_period": "6787.0916303823",
      "mean_motion": "0.000145221",
      "angular_momentum": "0.013202656",
      "ascending_node_distance": "0.32816",
      "descending_node_distance": "2.8741",
      "longitude": "35.4440798",
      "latitude": "28.1074538",
      "minimum_orbit_intersection_distance": "0.36251399",
      "last_perihelion_passage": "2020-Jul-03.6788980444"
    },
    "physical_parameters": {
      "GM": "n.a.",
      "radius": "n.a.",
      "M1": "12.1",
      "M2": "n.a.",
      "k1": "12.25",
      "k2": "n.a.",
      "PHCOF": "n.a."
    }
  },{
    "name": "Catalina (C/2013 US10)",
    "Rec #": "90004223 (+COV)",
    "solution_date": "2021-Apr-15_23:32:42",
    "observation_count": "4396 (2013-2017)",
    "epoch": "2457349.5 (2015-Nov-23.0000000 TDB)",
    "orbital_elements": {
      "residual_rms": "n.a.",
      "eccentricity": 1.000280111752132,
      "perihelion_distance": 0.8229694432560084,
      "time_of_perihelion_passage": 2457342.2216403401,
      "longitude_of_ascending_node": 186.1445801306016,
      "argument_of_perihelion": 340.3587540701534,
      "inclination": 148.8782970590471,
      "semi_major_axis": -2938.003982310757,
      "NK": 4.6142,
      "NM": 2.15,
      "NN": 5.093,
      "R0": 2.808,
      "mean_anomaly": 4.5046288727995e-5,
      "aphelion_distance": 9.999999E99,
      "orbital_period": 9.999999E99,
      "mean_motion": 6.189e-6,
      "angular_momentum": 0.022070832,
      "ascending_node_distance": 0.84763,
      "descending_node_distance": 28.42121,
      "longitude": 203.1339534,
      "latitude": -10.0048288,
      "minimum_orbit_intersection_distance": 0.13859899,
      "last_perihelion_passage": "2015-Nov-15.7216403401"
    },
    "physical_parameters": {
      "GM": "n.a.",
      "radius": "n.a.",
      "M1": 8.4,
      "M2": "n.a.",
      "k1": 6.5,
      "k2": "n.a.",
      "PHCOF": "n.a."
    },
    "non_gravitational_force_model": {
      "AMRAT": 0,
      "DT": 0,
      "A1": 7.573968172073e-9,
      "A2": 6.249690894037e-11,
      "A3": 1.621151342988e-10
    },
    "standard_model": {
      "ALN": 0.1112620426
    }
Mercury is the smallest planet in our solar system and nearest to the Sun, Mercury is only slightly larger than Earth's Moon. From the surface of Mercury, the Sun would appear more than three times as large as it does when viewed from Earth, and the sunlight would be as much as seven times brighter.
Mercury's surface temperatures are both extremely hot and cold. Because the planet is so close to the Sun, day temperatures can reach highs of 800°F (430°C). Without an atmosphere to retain that heat at night, temperatures can dip as low as -290°F (-180°C).
Despite its proximity to the Sun, Mercury is not the hottest planet in our solar system  that title belongs to nearby Venus, thanks to its dense atmosphere. But Mercury is the fastest planet, zipping around the Sun every 88 Earth days.
With a radius of 1,516 miles (2,440 kilometers), Mercury is a little more than 1/3 the width of Earth. If Earth were the size of a nickel, Mercury would be about as big as a blueberry.
From an average distance of 36 million miles (58 million kilometers), Mercury is 0.4 astronomical units away from the Sun. One astronomical unit (abbreviated as AU), is the distance from the Sun to Earth. From this distance, it takes sunlight 3.2 minutes to travel from the Sun to Mercury.
Mercury spins slowly on its axis and completes one rotation every 59 Earth days. But when Mercury is moving fastest in its elliptical orbit around the Sun (and it is closest to the Sun), each rotation is not accompanied by sunrise and sunset like it is on most other planets. The morning Sun appears to rise briefly, set, and rise again from some parts of the planet's surface. The same thing happens in reverse at sunset for other parts of the surface. One Mercury solar day (one full day-night cycle) equals 176 Earth days just over two years on Mercury.
Mercury is the second densest planet, after Earth. It has a large metallic core with a radius of about 1,289 miles (2,074 kilometers), about 85% of the planet's radius. There is evidence that it is partly molten or liquid. Mercury's outer shell, comparable to Earth's outer shell (called the mantle and crust), is only about 400 kilometers (250 miles) thick.
Mercury's surface resembles that of Earth's Moon, scarred by many impact craters resulting from collisions with meteoroids and comets. Craters and features on Mercury are named after famous deceased artists, musicians, or authors, including children's author Dr. Seuss and dance pioneer Alvin Ailey.
Very large impact basins, including Caloris (960 miles or 1,550 kilometers in diameter) and Rachmaninoff (190 miles, or 306 kilometers in diameter), were created by asteroid impacts on the planet's surface early in the solar system's history. While there are large areas of smooth terrain, there are also cliffs, some hundreds of miles long and soaring up to a mile high. They rose as the planet's interior cooled and contracted over the billions of years since Mercury formed.
Most of Mercury's surface would appear greyish-brown to the human eye. The bright streaks are called "crater rays." They are formed when an asteroid or comet strikes the surface. The tremendous amount of energy that is released in such an impact digs a big hole in the ground, and also crushes a huge amount of rock under the point of impact. Some of this crushed material is thrown far from the crater and then falls to the surface, forming the rays. Fine particles of crushed rock are more reflective than large pieces, so the rays look brighter. The space environment dust impacts and solar-wind particles causes the rays to darken with time.
Temperatures on Mercury are extreme. During the day, temperatures on the surface can reach 800 degrees Fahrenheit (430 degrees Celsius). Because the planet has no atmosphere to retain that heat, nighttime temperatures on the surface can drop to minus 290 degrees Fahrenheit (minus 180 degrees Celsius).
Mercury may have water ice at its north and south poles inside deep craters, but only in regions in permanent shadows. In those shadows, it could be cold enough to preserve water ice despite the high temperatures on sunlit parts of the planet.

Venus is the second planet from the Sun, and Earth's closest planetary neighbor. Venus is the third brightest object in the sky after the Sun and Moon. Venus spins slowly in the opposite direction from most planets.
Venus is similar in structure and size to Earth, and is sometimes called Earth's evil twin. Its thick atmosphere traps heat in a runaway greenhouse effect, making it the hottest planet in our solar system with surface temperatures hot enough to melt lead. Below the dense, persistent clouds, the surface has volcanoes and deformed mountains.
Venus orbits the Sun from an average distance of 67 million miles (108 million kilometers), or 0.72 astronomical units. One astronomical unit (abbreviated as AU), is the distance from the Sun to Earth. From this distance, it takes sunlight about six minutes to travel from the Sun to Venus.
Earth's nearness to Venus is a matter of perspective. The planet is nearly as big around as Earth. Its diameter at its equator is about 7,521 miles (12,104 kilometers), versus 7,926 miles (12,756 kilometers) for Earth. From Earth, Venus is the brightest object in the night sky after our own Moon. The ancients, therefore, gave it great importance in their cultures, even thinking it was two objects: a morning star and an evening star. That’s where the trick of perspective comes in.
Spending a day on Venus would be quite a disorienting experience - that is, if your spacecraft or spacesuit could protect you from temperatures in the range of 900 degrees Fahrenheit (475 Celsius). For one thing, your “day” would be 243 Earth days long – longer even than a Venus year (one trip around the Sun), which takes only 225 Earth days. For another, because of the planet's extremely slow rotation, sunrise to sunset would take 117 Earth days. And by the way, the Sun would rise in the west and set in the east, because Venus spins backward compared to Earth.
While you’re waiting, don’t expect any seasonal relief from the unrelenting temperatures. On Earth, with its spin axis tilted by about 23 degrees, we experience summer when our part of the planet (our hemisphere) receives the Sun’s rays more directly – a result of that tilt. In winter, the tilt means the rays are less direct. No such luck on Venus: Its very slight tilt is only three degrees, which is too little to produce noticeable seasons.
If we could slice Venus and Earth in half, pole to pole, and place them side by side, they would look remarkably similar. Each planet has an iron core enveloped by a hot-rock mantle; the thinnest of skins forms a rocky, exterior crust. On both planets, this thin skin changes form and sometimes erupts into volcanoes in response to the ebb and flow of heat and pressure deep beneath.
On Earth, the slow movement of continents over thousands and millions of years reshapes the surface, a process known as “plate tectonics.” Something similar might have happened on Venus early in its history. Today a key element of this process could be operating: subduction, or the sliding of one continental “plate” beneath another, which can also trigger volcanoes. Subduction is believed to be the first step in creating plate tectonics.


While Earth is only the fifth largest planet in the solar system, it is the only world in our solar system with liquid water on the surface. Just slightly larger than nearby Venus, Earth is the biggest of the four planets closest to the Sun, all of which are made of rock and metal.
With an equatorial diameter of 7926 miles (12,760 kilometers), 
Earth is the biggest of the terrestrial planets and the fifth largest planet in our solar system.
From an average distance of 93 million miles (150 million kilometers), Earth is exactly one astronomical unit away from the Sun because one astronomical unit (abbreviated as AU), is the distance from the Sun to Earth. This unit provides an easy way to quickly compare planets' distances from the Sun.
It takes about eight minutes for light from the Sun to reach our planet.
As Earth orbits the Sun, it completes one rotation every 23.9 hours. It takes 365.25 days to complete one trip around the Sun. That extra quarter of a day presents a challenge to our calendar system, which counts one year as 365 days. To keep our yearly calendars consistent with our orbit around the Sun, every four years we add one day. That day is called a leap day, and the year it's added to is called a leap year.
Earth's axis of rotation is tilted 23.4 degrees with respect to the plane of Earth's orbit around the Sun. This tilt causes our yearly cycle of seasons. During part of the year, the northern hemisphere is tilted toward the Sun, and the southern hemisphere is tilted away. With the Sun higher in the sky, solar heating is greater in the north producing summer there. Less direct solar heating produces winter in the south. Six months later, the situation is reversed. When spring and fall begin, both hemispheres receive roughly equal amounts of heat from the Sun.
Earth is composed of four main layers, starting with an inner core at the planet's center, enveloped by the outer core, mantle, and crust.
The inner core is a solid sphere made of iron and nickel metals about 759 miles (1,221 kilometers) in radius. There the temperature is as high as 9,800 degrees Fahrenheit (5,400 degrees Celsius). Surrounding the inner core is the outer core. This layer is about 1,400 miles (2,300 kilometers) thick, made of iron and nickel fluids.
In between the outer core and crust is the mantle, the thickest layer. This hot, viscous mixture of molten rock is about 1,800 miles (2,900 kilometers) thick and has the consistency of caramel. The outermost layer, Earth's crust, goes about 19 miles (30 kilometers) deep on average on land. At the bottom of the ocean, the crust is thinner and extends about 3 miles (5 kilometers) from the seafloor to the top of the mantle.
Near the surface, Earth has an atmosphere that consists of 78% nitrogen, 21% oxygen, and 1% other gases such as argon, carbon dioxide, and neon. The atmosphere affects Earth's long-term climate and short-term local weather and shields us from much of the harmful radiation coming from the Sun. It also protects us from meteoroids, most of which burn up in the atmosphere, seen as meteors in the night sky, before they can strike the surface as meteorites.

Mars is one of the most explored bodies in our solar system, and it's the only planet where we've sent rovers to roam the alien landscape. NASA missions have found lots of evidence that Mars was much wetter and warmer, with a thicker atmosphere, billions of years ago.
Mars was named by the Romans for their god of war because its reddish color was reminiscent of blood. The Egyptians called it "Her Desher," meaning "the red one."
Mars has two small moons, Phobos and Deimos, that may be captured asteroids. They're potato-shaped because they have too little mass for gravity to make them spherical.
The moons get their names from the horses that pulled the chariot of the Greek god of war, Ares.
Even today, it is frequently called the "Red Planet" because iron minerals in the Martian dirt oxidize, or rust, causing the surface to look red.
With a radius of 2,106 miles (3,390 kilometers), Mars is about half the size of Earth. If Earth were the size of a nickel, Mars would be about as big as a raspberry.
From an average distance of 142 million miles (228 million kilometers), Mars is 1.5 astronomical units away from the Sun. One astronomical unit (abbreviated as AU), is the distance from the Sun to Earth. From this distance, it takes sunlight 13 minutes to travel from the Sun to Mars.
Mars has a dense core at its center between 930 and 1,300 miles (1,500 to 2,100 kilometers) in radius. It's made of iron, nickel, and sulfur. Surrounding the core is a rocky mantle between 770 and 1,170 miles (1,240 to 1,880 kilometers) thick, and above that, a crust made of iron, magnesium, aluminum, calcium, and potassium. This crust is between 6 and 30 miles (10 to 50 kilometers) deep.
Mars has a thin atmosphere made up mostly of carbon dioxide, nitrogen, and argon gases. To our eyes, the sky would be hazy and red because of suspended dust instead of the familiar blue tint we see on Earth. Mars' sparse atmosphere doesn't offer much protection from impacts by such objects as meteorites, asteroids, and comets.
The temperature on Mars can be as high as 70 degrees Fahrenheit (20 degrees Celsius) or as low as about -225 degrees Fahrenheit (-153 degrees Celsius). And because the atmosphere is so thin, heat from the Sun easily escapes this planet. If you were to stand on the surface of Mars on the equator at noon, it would feel like spring at your feet (75 degrees Fahrenheit or 24 degrees Celsius) and winter at your head (32 degrees Fahrenheit or 0 degrees Celsius).

Jupiter's signature stripes and swirls are actually cold, windy clouds of ammonia and water, floating in an atmosphere of hydrogen and helium. The dark orange stripes are called belts, while the lighter bands are called zones, and they flow east and west in opposite directions. Jupiter’s iconic Great Red Spot is a giant storm bigger than Earth that has raged for hundreds of years.The king of planets was named for Jupiter, king of the gods in Roman mythology. Most of its moons are also named for mythological characters, figures associated with Jupiter or his Greek counterpart, Zeus.
With a radius of 43,440.7 miles (69,911 kilometers), Jupiter is 11 times wider than Earth. If Earth were the size of a grape, Jupiter would be about as big as a basketball.
From an average distance of 484 million miles (778 million kilometers), Jupiter is 5.2 astronomical units away from the Sun. One astronomical unit (abbreviated as AU), is the distance from the Sun to Earth. From this distance, it takes sunlight 43 minutes to travel from the Sun to Jupiter.
Jupiter has the shortest day in the solar system. One day on Jupiter takes only about 10 hours (the time it takes for Jupiter to rotate or spin around once), and Jupiter makes a complete orbit around the Sun (a year in Jovian time) in about 12 Earth years (4,333 Earth days).
Jupiter has 95 moons that are officially recognized by the International Astronomical Union. The four largest moons – Io, Europa, Ganymede, and Callisto – were first observed by the astronomer Galileo Galilei in 1610 using an early version of the telescope. These four moons are known today as the Galilean satellites, and they're some of the most fascinating destinations in our solar system.
The composition of Jupiter is similar to that of the Sun – mostly hydrogen and helium. Deep in the atmosphere, pressure and temperature increase, compressing the hydrogen gas into a liquid. This gives Jupiter the largest ocean in the solar system – an ocean made of hydrogen instead of water. Scientists think that, at depths perhaps halfway to the planet's center, the pressure becomes so great that electrons are squeezed off the hydrogen atoms, making the liquid electrically conducting like metal. Jupiter's fast rotation is thought to drive electrical currents in this region, with the spinning of the liquid metallic hydrogen acting like a dynamo, generating the planet's powerful magnetic field.
The vivid colors you see in thick bands across Jupiter may be plumes of sulfur and phosphorus-containing gases rising from the planet's warmer interior. Jupiter's fast rotation – spinning once every 10 hours – creates strong jet streams, separating its clouds into dark belts and bright zones across long stretches.
With no solid surface to slow them down, Jupiter's spots can persist for many years. Stormy Jupiter is swept by over a dozen prevailing winds, some reaching up to 335 miles per hour (539 kilometers per hour) at the equator. The Great Red Spot, a swirling oval of clouds twice as wide as Earth, has been observed on the giant planet for more than 300 years. More recently, three smaller ovals merged to form the Little Red Spot, about half the size of its larger cousin.

Saturn is the sixth planet from the Sun, and the second-largest planet in our solar system.
Like fellow gas giant Jupiter, Saturn is a massive ball made mostly of hydrogen and helium. Saturn is not the only planet to have rings, but none are as spectacular or as complex as Saturn's. Saturn also has dozens of moons.
From the jets of water that spray from Saturn's moon Enceladus to the methane lakes on smoggy Titan, the Saturn system is a rich source of scientific discovery and still holds many mysteries.
With an equatorial diameter of about 74,897 miles (120,500 kilometers), Saturn is 9 times wider than Earth. If Earth were the size of a nickel, Saturn would be about as big as a volleyball.
From an average distance of 886 million miles (1.4 billion kilometers), Saturn is 9.5 astronomical units away from the Sun. One astronomical unit (abbreviated as AU), is the distance from the Sun to Earth. From this distance, it takes sunlight 80 minutes to travel from the Sun to Saturn.
Saturn has the second-shortest day in the solar system. One day on Saturn takes only 10.7 hours (the time it takes for Saturn to rotate or spin around once), and Saturn makes a complete orbit around the Sun (a year in Saturnian time) in about 29.4 Earth years (10,756 Earth days).
Saturn is home to a vast array of intriguing and unique worlds. From the haze-shrouded surface of Titan to crater-riddled Phoebe, each of Saturn's moons tells another piece of the story surrounding the Saturn system. As of June 8, 2023, Saturn has 146 moons in its orbit, with others continually awaiting confirmation of their discovery and official naming by the International Astronomical Union (IAU).
Saturn's rings are thought to be pieces of comets, asteroids, or shattered moons that broke up before they reached the planet, torn apart by Saturn's powerful gravity. They are made of billions of small chunks of ice and rock coated with other materials such as dust. The ring particles mostly range from tiny, dust-sized icy grains to chunks as big as a house. A few particles are as large as mountains. The rings would look mostly white if you looked at them from the cloud tops of Saturn, and interestingly, each ring orbits at a different speed around the planet.
Saturn is blanketed with clouds that appear as faint stripes, jet streams, and storms. The planet is many different shades of yellow, brown, and gray.
Winds in the upper atmosphere reach 1,600 feet per second (500 meters per second) in the equatorial region. In contrast, the strongest hurricane-force winds on Earth top out at about 360 feet per second (110 meters per second). And the pressure – the same kind you feel when you dive deep underwater – is so powerful it squeezes gas into a liquid.

Uranus is the seventh planet from the Sun, and it has the third largest diameter of planets in our solar system. Uranus appears to spin sideways.
Uranus is a very cold and windy world. The ice giant is surrounded by 13 faint rings and 28 small moons. Uranus rotates at a nearly 90-degree angle from the plane of its orbit. This unique tilt makes Uranus appear to spin sideways, orbiting the Sun like a rolling ball.
Uranus was the first planet found with the aid of a telescope. It was discovered in 1781 by astronomer William Herschel, although he originally thought it was either a comet or a star. It was two years later that the object was universally accepted as a new planet, in part because of observations by astronomer Johann Elert Bode.
With an equatorial diameter of 31,763 miles (51,118 kilometers), Uranus is four times wider than Earth. If Earth was the size of a nickel, Uranus would be about as big as a softball.
From an average distance of 1.8 billion miles (2.9 billion kilometers), Uranus is about 19 astronomical units away from the Sun. One astronomical unit (abbreviated as AU), is the distance from the Sun to Earth. From this distance, it takes sunlight 2 hours and 40 minutes to travel from the Sun to Uranus.
One day on Uranus takes about 17 hours. This is the amount of time it takes Uranus to rotate, or spin once around its axis. Uranus makes a complete orbit around the Sun (a year in Uranian time) in about 84 Earth years (30,687 Earth days).
Uranus is the only planet whose equator is nearly at a right angle to its orbit, with a tilt of 97.77 degrees. This may be the result of a collision with an Earth-sized object long ago. This unique tilt causes Uranus to have the most extreme seasons in the solar system. For nearly a quarter of each Uranian year, the Sun shines directly over each pole, plunging the other half of the planet into a 21-year-long, dark winter.
Uranus has two sets of rings. The inner system of nine rings consists mostly of narrow, dark grey rings. There are two outer rings: the innermost one is reddish like dusty rings elsewhere in the solar system, and the outer ring is blue like Saturn's E ring.
Uranus is one of two ice giants in the outer solar system (the other is Neptune). Most (80% or more) of the planet's mass is made up of a hot dense fluid of "icy" materials – water, methane, and ammonia – above a small rocky core. Near the core, it heats up to 9,000 degrees Fahrenheit (4,982 degrees Celsius).
As an ice giant, Uranus doesn’t have a true surface. The planet is mostly swirling fluids. While a spacecraft would have nowhere to land on Uranus, it wouldn’t be able to fly through its atmosphere unscathed either. The extreme pressures and temperatures would destroy a metal spacecraft.

Neptune is the eighth and most distant planet in our solar system.
Dark, cold, and whipped by supersonic winds, ice giant Neptune is more than 30 times as far from the Sun as Earth. Neptune is the only planet in our solar system not visible to the naked eye. In 2011 Neptune completed its first 165-year orbit since its discovery in 1846.
Neptune is so far from the Sun that high noon on the big blue planet would seem like dim twilight to us. The warm light we see here on our home planet is roughly 900 times as bright as sunlight on Neptune.
With an equatorial diameter of 30,775 miles (49,528 kilometers), Neptune is about four times wider than Earth. If Earth were the size of a nickel, Neptune would be about as big as a baseball.
From an average distance of 2.8 billion miles (4.5 billion kilometers), Neptune is 30 astronomical units away from the Sun. One astronomical unit (abbreviated as AU), is the distance from the Sun to Earth. From this distance, it takes sunlight 4 hours to travel from the Sun to Neptune.
One day on Neptune takes about 16 hours (the time it takes for Neptune to rotate or spin once). And Neptune makes a complete orbit around the Sun (a year in Neptunian time) in about 165 Earth years (60,190 Earth days).
Neptune has 16 known moons. Neptune's largest moon Triton was discovered on Oct. 10, 1846, by William Lassell, just 17 days after Johann Gottfried Galle discovered the planet. Since Neptune was named for the Roman god of the sea, its moons are named for various lesser sea gods and nymphs in Greek mythology.
Neptune has at least five main rings and four prominent ring arcs that we know of so far. Starting near the planet and moving outward, the main rings are named Galle, Leverrier, Lassell, Arago, and Adams. The rings are thought to be relatively young and short-lived.
Neptune is one of two ice giants in the outer solar system (the other is Uranus). Most (80% or more) of the planet's mass is made up of a hot dense fluid of "icy" materials – water, methane, and ammonia – above a small, rocky core. Of the giant planets, Neptune is the densest.
Neptune does not have a solid surface. Its atmosphere (made up mostly of hydrogen, helium, and methane) extends to great depths, gradually merging into water and other melted ices over a heavier, solid core with about the same mass as Earth.

Pluto and other dwarf planets are a lot like regular planets. So what’s the big difference? The International Astronomical Union (IAU), a world organization of astronomers, came up with the definition of a planet in 2006. According to the IAU, a planet must do three things:
Orbit its host star (In our solar system that’s the Sun).
Be mostly round. Be big enough that its gravity cleared away any other objects of similar size near its orbit around the Sun.
Pluto is by far the most famous dwarf planet. Discovered by Clyde Tombaugh in 1930, Pluto was long considered our solar system's ninth planet. But after other astronomers found similar intriguing worlds deeper in the distant Kuiper Belt – the IAU reclassified Pluto as a dwarf planet in 2006. 
There was widespread outrage on behalf of the demoted planet. Textbooks were updated, and the internet spawned memes with Pluto going through a range of emotions, from anger to loneliness.
On July 14, 2015, NASA’s New Horizons spacecraft made its historic flight through the Pluto system – providing the first close-up images of Pluto and its moons and collecting other data that has transformed our understanding of these mysterious worlds on the solar system’s outer frontier.
Dwarf planet Ceres is closer to home. Ceres is the largest object in the asteroid belt between Mars and Jupiter, and it's the only dwarf planet located in the inner solar system. Like Pluto, Ceres also was once classified as a planet. Ceres was the first dwarf planet to be visited by a spacecraft – NASA’s Dawn mission. 

<<<<<<< HEAD:roberta2.py

Our Moon shares a name with all moons simply because people didn't know other moons existed until Galileo Galilei discovered four moons orbiting Jupiter in 1610. In Latin, the Moon was called Luna, which is the main adjective for all things Moon-related: lunar.
With a radius of about 1,080 miles (1,740 kilometers), the Moon is less than a third of the width of Earth. If Earth were the size of a nickel, the Moon would be about as big as a coffee bean.
The Moon is an average of 238,855 miles (384,400 kilometers) away. That means 30 Earth-sized planets could fit in between Earth and the Moon.
The Moon is slowly moving away from Earth, getting about an inch farther away each year.
The Moon is rotating at the same rate that it revolves around Earth (called synchronous rotation), so the same hemisphere faces Earth all the time. Some people call the far side – the hemisphere we never see from Earth – the "dark side", but that's misleading.
As the Moon orbits Earth, different parts are in sunlight or darkness at different times. The changing illumination is why, from our perspective, the Moon goes through phases. During a "full moon," the hemisphere of the Moon we can see from Earth is fully illuminated by the Sun. And a "new moon" occurs when the far side of the Moon has full sunlight, and the side facing us is having its night.
The Moon makes a complete orbit around Earth in 27 Earth days and rotates or spins at that same rate, or in that same amount of time. Because Earth is moving as well – rotating on its axis as it orbits the Sun – from our perspective, the Moon appears to orbit us every 29 days.
The Moon likely formed after a Mars-sized body collided with Earth several billion years ago.
The resulting debris from both Earth and the impactor accumulated to form our natural satellite 239,000 miles (384,000 kilometers) away. The newly formed Moon was in a molten state, but within about 100 million years, most of the global "magma ocean" had crystallized, with less-dense rocks floating upward and eventually forming the lunar crust.
Over billions of years, these impacts have ground up the surface of the Moon into fragments ranging from huge boulders to powder. Nearly the entire Moon is covered by a rubble pile of charcoal-gray, powdery dust, and rocky debris called the lunar regolith. Beneath is a region of fractured bedrock referred to as the megaregolith.
In October 2020, NASA’s Stratospheric Observatory for Infrared Astronomy (SOFIA) confirmed, for the first time, water on the sunlit surface of the Moon. This discovery indicates that water may be distributed across the lunar surface, and not limited to cold, shadowed places. SOFIA detected water molecules (H2O) in Clavius Crater, one of the largest craters visible from Earth, located in the Moon’s southern hemisphere.

Halley's Comet is a periodic comet, meaning it orbits the sun and returns to the inner solar system on a regular basis. It takes 75–76 years to orbit the sun and return to Earth, where it can be seen by the naked eye. 
Halley's Comet is about 11 kilometers in diameter, making it larger than 99% of asteroids and comparable in size to Boston. 
Halley's Comet is made up of water and gases like methane, ammonia, and carbon dioxide. 
As Halley's Comet approaches the sun, the ice and gases inside it expand and form a tail that can look like a shooting star. Comets have two tails, a dust tail and an ion (gas) tail. 
Halley's Comet was the first comet whose return was predicted, proving that some comets are part of our solar system. Edmond Halley predicted its return in 1758 after showing that comets seen in 1531, 1607, and 1682 were actually the same comet. 
Halley's Comet's next appearance is predicted to be in July 2061

Apophis is about 375 meters wide, roughly the size of a cruise liner. 
Apophis is an Aten asteroid, which means it has an orbit with a semi-major axis of less than one astronomical unit. 
Apophis's orbit takes it around the sun every 323,513 days, and it crosses Earth's orbit twice per revolution. 
On April 13, 2029, Apophis will pass within 32,000 kilometers of Earth's surface. It will be visible to the naked eye for about two billion people in parts of Asia, Europe, and Africa. 
In 2036, Apophis will pass more than 49 million kilometers from Earth. In 2116, Apophis could pass as close as 150,000 kilometers from Earth. 
NASA says that Apophis poses no threat to Earth for at least the next century, however they are constantly keeping an eye on it, in case its orbit changes.
Apophis was discovered in 2004. 
Apophis is also known as Apep, the great serpent and enemy of the Egyptian sun god Ra. 

The asteroid belt is a region within the solar system occupied by asteroids that are sparsely held together by gravity and occupying a region taking the shape of a gradient ring orbiting the Sun. Asteroids are small rocky bodies sometimes composed of iron and nickel, which orbit the Sun. The asteroid belt exists between the orbits of Mars and Jupiter, between 330 million and 480 million kilometers from the Sun.
Location: The asteroid belt is located between the orbits of Mars and Jupiter 
Size: The asteroid belt is a torus-shaped or disc-shaped region that contains millions or billions of asteroids 
Composition: The asteroids in the belt are made up of carbon-rich materials and ices 
Formation: The asteroid belt formed about 4.5 billion years ago, along with the rest of the solar system 
Asteroids: Over 600,000 asteroids in the belt have been identified and named 
Ceres: Ceres is the largest known asteroid at 620 miles across 
Density: The asteroid belt is not as densely packed as fiction often portrays it 
Color: The asteroid belt is largely empty space, so there isn't really much color to see at all 
Resonances: The mean distances of the asteroids are not uniformly distributed but exhibit population depletions, or “gaps”

"What is the smallest planet in our solar system?": "Mercury is the smallest planet in our solar system.",
    "How does the size of Mercury compare to Earth's Moon?": "Mercury is only slightly larger than Earth's Moon.",
    "How much brighter is sunlight on Mercury compared to Earth?": "Sunlight on Mercury is up to seven times brighter than on Earth.",
    "What are the temperature extremes on Mercury?": "Daytime temperatures on Mercury can reach 800°F (430°C), while nighttime temperatures can drop to -290°F (-180°C).",
    "Why isn’t Mercury the hottest planet despite being closest to the Sun?": "Venus is the hottest planet due to its dense atmosphere, which traps heat more effectively than Mercury's lack of atmosphere.",
    "How long does it take Mercury to orbit the Sun?": "Mercury orbits the Sun every 88 Earth days.",
    "How long is a solar day on Mercury?": "A solar day on Mercury lasts 176 Earth days.",
    "What is the composition of Mercury's core?": "Mercury has a large metallic core, making up about 85% of its radius, and it is partly molten or liquid.",
    "What are the 'crater rays' on Mercury's surface?": "Crater rays are bright streaks formed when an asteroid or comet strikes Mercury’s surface, scattering reflective crushed material.",
    "Can Mercury have water ice?": "Yes, water ice may exist in deep craters at Mercury’s poles in regions that are permanently shadowed, despite the extreme temperatures on other parts of the planet.",
    "Which planet is the second from the Sun?": "Venus is the second planet from the Sun.",
    "What is Earth's closest planetary neighbor?": "Venus is Earth's closest planetary neighbor.",
    "What is the third brightest object in the sky after the Sun and the Moon?": "Venus is the third brightest object in the sky after the Sun and Moon.",
    "How does Venus spin compared to most planets?": "Venus spins slowly in the opposite direction from most planets.",
    "Why is Venus sometimes called Earth's evil twin?": "Venus is similar in structure and size to Earth, and its thick atmosphere makes it extremely hot, earning it the nickname 'Earth's evil twin.'",
    "What makes Venus the hottest planet in our solar system?": "Venus has a thick atmosphere that traps heat in a runaway greenhouse effect, making it the hottest planet in our solar system.",
    "What is Venus' average distance from the Sun?": "Venus orbits the Sun from an average distance of 67 million miles (108 million kilometers).",
    "How long does it take for sunlight to travel from the Sun to Venus?": "It takes sunlight about six minutes to travel from the Sun to Venus.",
    "How does Venus' size compare to Earth's?": "Venus' diameter is about 7,521 miles (12,104 kilometers), which is slightly smaller than Earth's diameter of 7,926 miles (12,756 kilometers).",
    "Why did the ancients think Venus was two different objects?": "Ancients thought Venus was two different objects, a morning star and an evening star, due to the way it appears in the sky at different times.",
    "What would it be like to spend a day on Venus?": "Spending a day on Venus would be disorienting, with extremely high temperatures around 900 degrees Fahrenheit (475 Celsius), and a day that lasts 243 Earth days.",
    "How long is a day on Venus compared to a Venusian year?": "A day on Venus lasts 243 Earth days, which is longer than a year on Venus, which takes only 225 Earth days.",
    "Why does the Sun rise in the west and set in the east on Venus?": "On Venus, the Sun rises in the west and sets in the east because Venus spins in the opposite direction compared to Earth.",
    "Why doesn't Venus have seasons like Earth?": "Venus doesn’t have noticeable seasons because its axis tilt is only three degrees, unlike Earth's 23-degree tilt.",
    "How do Venus and Earth compare in structure?": "Venus and Earth have similar structures, with iron cores surrounded by hot rock mantles and a thin rocky crust.",
    "What is plate tectonics and how does it relate to Venus?": "Plate tectonics is the process of continents moving and reshaping the surface of a planet. Venus may have experienced subduction, a key part of tectonics.",
    "How does Earth's tilt affect its seasons?": "Earth's axis is tilted 23.4 degrees, causing different parts of the planet to receive varying amounts of sunlight, which leads to the changing seasons.",
    "What are the main layers of Earth?": "Earth is composed of four main layers: the inner core, outer core, mantle, and crust.",
    "How long does it take for light from the Sun to reach Earth?": "It takes about eight minutes for sunlight to travel from the Sun to Earth.",
    "What is the equatorial diameter of Earth?": "The equatorial diameter of Earth is 7,926 miles (12,760 kilometers).",
    "Why is Mars one of the most explored bodies in our solar system?": "Mars is one of the most explored bodies because NASA has sent multiple missions and rovers to study its surface, finding evidence of a wetter and warmer past.",
    "Why is Mars called the 'Red Planet'?": "Mars is called the 'Red Planet' because its surface is covered in iron oxide, or rust, which gives it a reddish appearance.",
    "How many moons does Mars have?": "Mars has two small moons, Phobos and Deimos, which are likely captured asteroids.",
    "How far is Mars from the Sun?": "Mars is about 142 million miles (228 million kilometers) from the Sun, which is 1.5 astronomical units (AU).",
    "How long does it take sunlight to reach Mars?": "It takes sunlight approximately 13 minutes to travel from the Sun to Mars.",
    "What is the core of Mars made of?": "Mars has a dense core made of iron, nickel, and sulfur, with a radius between 930 and 1,300 miles (1,500 to 2,100 kilometers).",
    "What is Mars' atmosphere made of?": "Mars' atmosphere is composed mainly of carbon dioxide, nitrogen, and argon gases.",
    "What is the range of temperatures on Mars?": "Temperatures on Mars can range from 70 degrees Fahrenheit (20 degrees Celsius) during the day to -225 degrees Fahrenheit (-153 degrees Celsius) at night.",
    "Why does Mars' sky appear hazy and red?": "Mars' sky appears hazy and red due to the suspended dust in the thin atmosphere, which scatters light differently than Earth's atmosphere.",
    "What is the size of Mars compared to Earth?": "Mars has a radius of 2,106 miles (3,390 kilometers), which is about half the size of Earth.",
    "What are Jupiter's stripes and swirls made of?": "Jupiter's stripes and swirls are made of cold, windy clouds of ammonia and water, floating in an atmosphere of hydrogen and helium.",
    "What is the Great Red Spot on Jupiter?": "The Great Red Spot is a massive storm on Jupiter that is larger than Earth and has been active for hundreds of years.",
    "How many moons does Jupiter have?": "Jupiter has 95 moons that are officially recognized by the International Astronomical Union.",
    "What are Jupiter's four largest moons called?": "Jupiter's four largest moons are called the Galilean satellites: Io, Europa, Ganymede, and Callisto.",
    "How big is Jupiter compared to Earth?": "Jupiter is 11 times wider than Earth, with a radius of 43,440.7 miles (69,911 kilometers).",
    "How long does it take for Jupiter to orbit the Sun?": "Jupiter takes about 12 Earth years (4,333 Earth days) to make a complete orbit around the Sun.",
    "What is Jupiter's atmosphere primarily made of?": "Jupiter's atmosphere is mostly made of hydrogen and helium.",
    "How fast do winds on Jupiter blow?": "Winds on Jupiter can reach speeds of up to 335 miles per hour (539 kilometers per hour) at the equator.",
    "Why do Jupiter's bands and belts flow in opposite directions?": "Jupiter's fast rotation creates strong jet streams, causing the dark belts and light zones to flow east and west in opposite directions.",
    "How long does one day on Jupiter last?": "One day on Jupiter lasts about 10 Earth hours, which is the time it takes for Jupiter to complete one rotation.",
    "What position does Saturn hold in the solar system?": "Saturn is the sixth planet from the Sun.",
    "What is Saturn primarily made of?": "Saturn is made mostly of hydrogen and helium.",
    "How do Saturn's rings compare to those of other planets?": "Saturn's rings are the most spectacular and complex of any planet in the solar system.",
    "What is unique about Saturn's moon Titan?": "Saturn's moon Titan has methane lakes and is covered by a thick, smoggy atmosphere.",
    "How wide is Saturn compared to Earth?": "Saturn is about 9 times wider than Earth, with an equatorial diameter of 74,897 miles (120,500 kilometers).",
    "How far is Saturn from the Sun?": "Saturn is about 886 million miles (1.4 billion kilometers) from the Sun, or 9.5 astronomical units.",
    "How long is a day on Saturn?": "A day on Saturn lasts only 10.7 hours.",
    "How many moons does Saturn have?": "As of June 8, 2023, Saturn has 146 moons.",
    "What are Saturn's rings made of?": "Saturn's rings are made of billions of small chunks of ice and rock, coated with materials like dust.",
    "How fast are the winds in Saturn's upper atmosphere?": "Winds in Saturn's upper atmosphere can reach speeds of up to 1,600 feet per second (500 meters per second).",
    "What position does Saturn hold in the solar system?": "Saturn is the sixth planet from the Sun.",
    "What is Saturn primarily made of?": "Saturn is made mostly of hydrogen and helium.",
    "What is the position of Uranus in the solar system?": "Uranus is the seventh planet from the Sun.",
    "How does Uranus' rotation differ from other planets?": "Uranus appears to spin sideways, rotating at a nearly 90-degree angle from the plane of its orbit.",
    "How many rings and moons does Uranus have?": "Uranus is surrounded by 13 faint rings and 28 small moons.",
    "Who discovered Uranus and when?": "Uranus was discovered in 1781 by astronomer William Herschel.",
    "What is the equatorial diameter of Uranus compared to Earth?": "Uranus has an equatorial diameter of 31,763 miles (51,118 kilometers), which is four times wider than Earth.",
    "How far is Uranus from the Sun?": "Uranus is 1.8 billion miles (2.9 billion kilometers) from the Sun, or about 19 astronomical units.",
    "How long is a day on Uranus?": "A day on Uranus takes about 17 hours.",
    "How long does Uranus take to orbit the Sun?": "Uranus takes about 84 Earth years (30,687 Earth days) to orbit the Sun.",
    "What causes Uranus' extreme seasons?": "Uranus' unique tilt of 97.77 degrees causes it to have the most extreme seasons in the solar system.",
    "What is Uranus made of?": "Most of Uranus' mass is made up of a hot dense fluid of water, methane, and ammonia above a small rocky core.",
    "What is Neptune's position in the solar system?": "Neptune is the eighth and most distant planet in our solar system.",
    "How does Neptune's distance from the Sun compare to Earth's?": "Neptune is more than 30 times as far from the Sun as Earth.",
    "Why can't Neptune be seen with the naked eye?": "Neptune is the only planet in our solar system not visible to the naked eye due to its great distance from the Sun.",
    "How long does it take Neptune to complete one orbit around the Sun?": "Neptune completes one orbit around the Sun in about 165 Earth years.",
    "What is Neptune's equatorial diameter compared to Earth?": "Neptune has an equatorial diameter of 30,775 miles (49,528 kilometers), which is about four times wider than Earth.",
    "How far is Neptune from the Sun?": "Neptune is 2.8 billion miles (4.5 billion kilometers) from the Sun, or 30 astronomical units away.",
    "How long does it take for sunlight to reach Neptune?": "It takes sunlight about 4 hours to travel from the Sun to Neptune.",
    "What are the names of Neptune's rings?": "Neptune's rings are named Galle, Leverrier, Lassell, Arago, and Adams.",
    "What is Neptune primarily made of?": "Neptune is made of a hot dense fluid of icy materials like water, methane, and ammonia above a small, rocky core.",
    "How long is a day on Neptune?": "A day on Neptune takes about 16 hours.",
    "What are the three criteria that define a planet according to the IAU?": "A planet must orbit its host star, be mostly round, and have cleared its orbit of other objects of similar size.",
    "Why was Pluto reclassified as a dwarf planet?": "Pluto was reclassified as a dwarf planet in 2006 because it does not clear other objects of similar size from its orbit.",
    "Who discovered Pluto and when?": "Pluto was discovered by Clyde Tombaugh in 1930.",
    "What caused outrage when Pluto was reclassified as a dwarf planet?": "The reclassification of Pluto as a dwarf planet caused widespread outrage because it had long been considered the ninth planet.",
    "When did NASA’s New Horizons spacecraft make its historic flight through the Pluto system?": "NASA’s New Horizons spacecraft flew through the Pluto system on July 14, 2015.",
    "What did the New Horizons mission contribute to our understanding of Pluto?": "The New Horizons mission provided the first close-up images of Pluto and its moons, transforming our understanding of these distant worlds.",
    "What is Ceres, and where is it located?": "Ceres is the largest object in the asteroid belt between Mars and Jupiter, and it is the only dwarf planet located in the inner solar system.",
    "Was Ceres always considered a dwarf planet?": "No, like Pluto, Ceres was once classified as a planet before being reclassified as a dwarf planet.",
    "Which spacecraft was the first to visit Ceres?": "NASA’s Dawn mission was the first spacecraft to visit the dwarf planet Ceres.",
    "Where is Pluto located in the solar system?": "Pluto is located in the distant Kuiper Belt, beyond the orbit of Neptune.",
    "Why do all moons share the same name as our Moon?": "People didn't know other moons existed until Galileo Galilei discovered four moons orbiting Jupiter in 1610, so they called our Moon just 'the Moon.'",
    "What is the Latin name for the Moon, and how is it used?": "The Latin name for the Moon is 'Luna', and it is used as the adjective for Moon-related things, such as 'lunar.'",
    "How does the size of the Moon compare to Earth?": "The Moon's radius is about 1,080 miles (1,740 kilometers), which is less than a third of Earth's width.",
    "How far is the Moon from Earth?": "The Moon is an average of 238,855 miles (384,400 kilometers) away from Earth.",
    "How does the Moon's distance from Earth change over time?": "The Moon is slowly moving away from Earth, getting about an inch farther away each year.",
    "Why do we always see the same side of the Moon from Earth?": "The Moon rotates at the same rate that it revolves around Earth (synchronous rotation), so the same hemisphere always faces Earth.",
    "What causes the Moon's phases?": "The Moon's phases are caused by different parts of the Moon being illuminated by the Sun as it orbits Earth.",
    "How long does it take the Moon to orbit Earth?": "The Moon makes a complete orbit around Earth in 27 Earth days.",
    "How did the Moon form?": "The Moon likely formed after a Mars-sized body collided with Earth, and the resulting debris accumulated to form our natural satellite.",
    "What did NASA's SOFIA discover about the Moon in 2020?": "NASA's SOFIA confirmed the presence of water on the sunlit surface of the Moon in 2020, indicating water may be distributed across the lunar surface.",
    "What is a periodic comet?": "A periodic comet orbits the sun and returns to the inner solar system on a regular basis.",
    "How long does it take Halley's Comet to complete one orbit around the sun?": "It takes 75–76 years for Halley's Comet to orbit the sun and return to Earth.",
    "How large is Halley's Comet?": "Halley's Comet is about 11 kilometers in diameter, comparable in size to the city of Boston.",
    "What is Halley's Comet made of?": "Halley's Comet is made up of water and gases like methane, ammonia, and carbon dioxide.",
    "What happens to Halley's Comet as it approaches the sun?": "As Halley's Comet approaches the sun, the ice and gases inside it expand and form a tail.",
    "How many tails does Halley's Comet have?": "Halley's Comet has two tails: a dust tail and an ion (gas) tail.",
    "Who predicted the return of Halley's Comet?": "Edmond Halley predicted the return of Halley's Comet in 1758.",
    "When was it first proven that some comets are part of our solar system?": "It was first proven that some comets are part of our solar system when Edmond Halley showed that comets seen in 1531, 1607, and 1682 were actually the same comet.",
    "When will Halley's Comet next appear?": "Halley's Comet's next appearance is predicted to be in July 2061.",
    "How often can Halley's Comet be seen from Earth?": "Halley's Comet can be seen from Earth roughly every 75–76 years.",
    "How large is Apophis?": "Apophis is about 375 meters wide, roughly the size of a cruise liner.",
    "What is an Aten asteroid?": "An Aten asteroid has an orbit with a semi-major axis of less than one astronomical unit.",
    "How often does Apophis orbit the Sun?": "Apophis completes an orbit around the Sun every 323,513 days.",
    "When will Apophis make its close pass to Earth?": "On April 13, 2029, Apophis will pass within 32,000 kilometers of Earth's surface.",
    "How many people will be able to see Apophis when it passes by Earth in 2029?": "Apophis will be visible to about two billion people in parts of Asia, Europe, and Africa.",
    "How close will Apophis pass to Earth in 2036?": "In 2036, Apophis will pass more than 49 million kilometers from Earth.",
    "What is the closest Apophis could pass to Earth in 2116?": "Apophis could pass as close as 150,000 kilometers from Earth in 2116.",
    "Does Apophis pose any threat to Earth?": "NASA says that Apophis poses no threat to Earth for at least the next century.",
    "When was Apophis discovered?": "Apophis was discovered in 2004.",
    "What is the mythological origin of the name Apophis?": "Apophis is also known as Apep, the great serpent and enemy of the Egyptian sun god Ra.",
    "Where is the asteroid belt located in the solar system?": "The asteroid belt is located between the orbits of Mars and Jupiter.",
    "What shape does the asteroid belt take?": "The asteroid belt is a torus-shaped or disc-shaped region.",
    "What are asteroids in the asteroid belt made of?": "The asteroids are composed of carbon-rich materials, ices, and sometimes iron and nickel.",
    "How old is the asteroid belt?": "The asteroid belt formed about 4.5 billion years ago, along with the rest of the solar system.",
    "How many asteroids in the asteroid belt have been identified?": "Over 600,000 asteroids in the belt have been identified and named.",
    "What is the largest known asteroid in the asteroid belt?": "Ceres is the largest known asteroid, measuring 620 miles across.",
    "Is the asteroid belt densely packed?": "No, the asteroid belt is not as densely packed as often portrayed in fiction.",
    "What color is the asteroid belt?": "The asteroid belt is largely empty space, so there isn’t much color to see.",
    "What is the distance between the asteroid belt and the Sun?": "The asteroid belt exists between 330 million and 480 million kilometers from the Sun.",
    "What are resonances in the asteroid belt?": "The asteroid belt exhibits population depletions, or gaps, due to resonances in the mean distances of the asteroids."
"""

# Bucle de conversación
print("Chatbot: ¡Hola! Estoy aquí para responder tus preguntas sobre inteligencia artificial.")
while True:
    # Obtener entrada del usuario
    user_input = input("Tú: ")

    # Salir de la conversación si el usuario escribe "salir"
    if user_input.lower() in ["salir", "exit", "quitar"]:
        print("Chatbot: ¡Hasta luego!")
        break

    # Generar respuesta usando el modelo
    result = pipe(question=user_input, context=context)

    # Ajustar la longitud de la respuesta
    answer = result['answer']
    min_length = 5  # longitud mínima
    max_length = 250  # longitud máxima

    # Verificar la longitud de la respuesta
    if len(answer) < min_length:
        answer = "Lo siento, no puedo proporcionar una respuesta detallada en este momento."
    elif len(answer) > max_length:
        answer = answer[:max_length] + "..."

    # Imprimir la respuesta del modelo
    print("Chatbot:", answer)

# Save the fine-tuned model
trainer.save_model("./fine_tuned_Roberta")

