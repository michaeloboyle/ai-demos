#!/usr/bin/env python3
"""
Synthetic Defect Pattern Generation Script

Generates physics-based synthetic defect patterns for helmet QC analysis:
- Impact damage patterns with stress propagation
- Material degradation effects (UV, chemical, thermal)
- Manufacturing defects (delamination, void patterns, surface flaws)
- Wear pattern simulation based on usage scenarios
"""

import json
import os
import time
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Base directory for saving defect patterns
BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'assets', 'defect_patterns')

def generate_synthetic_defects():
    """Generate comprehensive synthetic defect patterns for QC analysis"""

    print("🔬 Generating Physics-Based Synthetic Defect Patterns...")

    defect_data = {}

    # 1. Impact Damage Patterns
    impact_patterns = generate_impact_damage_patterns()
    defect_data.update(impact_patterns)

    # 2. Material Degradation Effects
    degradation_patterns = generate_material_degradation()
    defect_data.update(degradation_patterns)

    # 3. Manufacturing Defects
    manufacturing_defects = generate_manufacturing_defects()
    defect_data.update(manufacturing_defects)

    # 4. Wear Pattern Simulation
    wear_patterns = generate_wear_patterns()
    defect_data.update(wear_patterns)

    # 5. Environmental Damage
    environmental_damage = generate_environmental_damage()
    defect_data.update(environmental_damage)

    return defect_data

def generate_impact_damage_patterns():
    """Generate realistic impact damage patterns with physics-based propagation"""

    print("  💥 Impact Damage Patterns...")

    impact_data = {}

    # Physics-based impact scenarios
    impact_scenarios = {
        'BALLISTIC_IMPACT_V50': {
            'title': 'Ballistic Impact - V50 Test Pattern',
            'description': 'Synthetic ballistic impact pattern at V50 velocity threshold',
            'impact_type': 'ballistic',
            'projectile': '17_grain_FSP',
            'velocity_mps': 650,
            'impact_location': {'x': 0.45, 'y': 0.35},  # Normalized coordinates
            'damage_radius': 15,  # mm
            'stress_pattern': 'radial_symmetric',
            'material_response': 'aramid_fiber_delamination'
        },
        'BLUNT_TRAUMA_IMPACT': {
            'title': 'Blunt Force Trauma - Drop Test Simulation',
            'description': 'Simulated drop impact from 2m height onto concrete',
            'impact_type': 'blunt_force',
            'impactor': 'concrete_surface',
            'impact_energy_j': 45,
            'contact_area': 1250,  # mm²
            'damage_pattern': 'compression_shear',
            'crack_propagation': 'stress_concentration'
        },
        'FRAGMENT_IMPACT_PATTERN': {
            'title': 'Fragment Impact - Multi-Point Damage',
            'description': 'Multiple fragment impacts from explosive proximity',
            'impact_type': 'fragmentation',
            'fragment_count': 7,
            'fragment_size_range': [2, 8],  # mm
            'impact_angles': [15, 30, 45, 60, 75],  # degrees
            'damage_distribution': 'clustered_random'
        }
    }

    for pattern_id, info in impact_scenarios.items():
        # Generate detailed defect geometry and physics
        defect_geometry = calculate_impact_damage_geometry(info)

        impact_data[pattern_id] = {
            'id': pattern_id,
            'title': info['title'],
            'source': 'Physics-Based Simulation',
            'description': info['description'],
            'defect_type': 'impact_damage',
            'impact_scenario': info,
            'damage_geometry': defect_geometry,
            'detectability': 'high',
            'criticality': 'critical',
            'inspection_method': 'visual_tactile',
            'generated_date': time.strftime('%Y-%m-%d'),
            'physics_based': True,
            'validation_status': 'synthetic_verified'
        }

        print(f"    ✅ {pattern_id}: {info['title']}")

    return impact_data

def generate_material_degradation():
    """Generate material degradation patterns from environmental exposure"""

    print("  🧪 Material Degradation Patterns...")

    degradation_data = {}

    degradation_scenarios = {
        'UV_POLYMER_DEGRADATION': {
            'title': 'UV Radiation Polymer Chain Scission',
            'description': 'Long-term UV exposure causing molecular chain breakdown',
            'degradation_type': 'photochemical',
            'exposure_duration_hours': 2000,
            'wavelength_nm': [280, 320],
            'affected_materials': ['polycarbonate_shell', 'aramid_fibers'],
            'degradation_pattern': 'surface_chalking_embrittlement'
        },
        'THERMAL_CYCLING_STRESS': {
            'title': 'Thermal Cycling Induced Microcracking',
            'description': 'Temperature cycling causing differential expansion stress',
            'degradation_type': 'thermal_mechanical',
            'temperature_range_c': [-40, 60],
            'cycle_count': 500,
            'stress_concentration': 'material_interfaces',
            'crack_initiation': 'thermal_expansion_mismatch'
        },
        'CHEMICAL_EXPOSURE_DEGRADATION': {
            'title': 'Chemical Agent Degradation Pattern',
            'description': 'Synthetic exposure to common field chemicals',
            'degradation_type': 'chemical',
            'exposure_agents': ['diesel_fuel', 'hydraulic_fluid', 'cleaning_solvents'],
            'exposure_duration_hours': 168,  # 1 week
            'degradation_effects': ['plasticizer_migration', 'surface_etching', 'color_change']
        }
    }

    for pattern_id, info in degradation_scenarios.items():
        # Calculate degradation progression
        degradation_model = model_material_degradation(info)

        degradation_data[pattern_id] = {
            'id': pattern_id,
            'title': info['title'],
            'source': 'Material Science Simulation',
            'description': info['description'],
            'defect_type': 'material_degradation',
            'degradation_scenario': info,
            'degradation_model': degradation_model,
            'detectability': 'medium',
            'criticality': 'major',
            'inspection_method': 'visual_spectroscopic',
            'generated_date': time.strftime('%Y-%m-%d'),
            'physics_based': True,
            'time_dependent': True
        }

        print(f"    ✅ {pattern_id}: {info['title']}")

    return degradation_data

def generate_manufacturing_defects():
    """Generate manufacturing defect patterns from production processes"""

    print("  🏭 Manufacturing Defect Patterns...")

    manufacturing_data = {}

    manufacturing_scenarios = {
        'RESIN_VOID_FORMATION': {
            'title': 'Resin Matrix Void Formation',
            'description': 'Air entrapment during resin transfer molding',
            'defect_type': 'void_formation',
            'process_stage': 'resin_transfer_molding',
            'void_size_distribution': 'lognormal',
            'void_density_per_cm2': 2.3,
            'formation_mechanism': 'air_entrapment_degassing'
        },
        'FIBER_MISALIGNMENT_PATTERN': {
            'title': 'Aramid Fiber Misalignment Defect',
            'description': 'Fiber bundle misalignment during layup process',
            'defect_type': 'fiber_orientation',
            'target_angle_degrees': 0,
            'actual_angle_degrees': 15,
            'affected_area_percent': 12,
            'strength_reduction_percent': 25
        },
        'DELAMINATION_INTERFACE': {
            'title': 'Inter-ply Delamination Pattern',
            'description': 'Adhesion failure between composite layers',
            'defect_type': 'delamination',
            'interface_location': 'ply_3_4_boundary',
            'delamination_area_mm2': 850,
            'propagation_direction': 'fiber_parallel',
            'detection_method': 'ultrasonic_c_scan'
        }
    }

    for pattern_id, info in manufacturing_scenarios.items():
        # Generate manufacturing defect geometry
        defect_characteristics = model_manufacturing_defect(info)

        manufacturing_data[pattern_id] = {
            'id': pattern_id,
            'title': info['title'],
            'source': 'Manufacturing Process Simulation',
            'description': info['description'],
            'defect_type': 'manufacturing_defect',
            'manufacturing_scenario': info,
            'defect_characteristics': defect_characteristics,
            'detectability': 'medium',
            'criticality': 'minor_to_major',
            'inspection_method': 'ndt_ultrasonic',
            'generated_date': time.strftime('%Y-%m-%d'),
            'process_related': True,
            'quality_control_stage': 'post_molding'
        }

        print(f"    ✅ {pattern_id}: {info['title']}")

    return manufacturing_data

def generate_wear_patterns():
    """Generate realistic wear patterns from usage scenarios"""

    print("  👤 Usage Wear Patterns...")

    wear_data = {}

    wear_scenarios = {
        'CHIN_STRAP_WEAR_PATTERN': {
            'title': 'Chin Strap Contact Wear',
            'description': 'Localized wear from chin strap friction over 200 use cycles',
            'wear_type': 'abrasive_contact',
            'contact_location': 'chin_strap_attachment',
            'use_cycles': 200,
            'contact_pressure_kpa': 15,
            'wear_depth_microns': 45,
            'wear_area_shape': 'elliptical'
        },
        'MOUNTING_RAIL_WEAR': {
            'title': 'NVG Mount Rail Wear Pattern',
            'description': 'Metal-on-composite wear from night vision device mounting',
            'wear_type': 'adhesive_abrasive',
            'mounting_cycles': 150,
            'contact_material': 'aluminum_7075',
            'wear_groove_depth': 0.3,  # mm
            'surface_roughness_increase': 'Ra_2.5_to_8.2'
        },
        'PADDING_COMPRESSION_SET': {
            'title': 'Interior Padding Compression Set',
            'description': 'Permanent deformation of interior foam padding',
            'wear_type': 'viscoelastic_deformation',
            'compression_cycles': 500,
            'max_compression_percent': 60,
            'recovery_time_hours': 24,
            'permanent_set_percent': 15
        }
    }

    for pattern_id, info in wear_scenarios.items():
        # Model wear progression
        wear_progression = model_wear_progression(info)

        wear_data[pattern_id] = {
            'id': pattern_id,
            'title': info['title'],
            'source': 'Usage Simulation Model',
            'description': info['description'],
            'defect_type': 'wear_degradation',
            'wear_scenario': info,
            'wear_progression': wear_progression,
            'detectability': 'high',
            'criticality': 'minor',
            'inspection_method': 'visual_dimensional',
            'generated_date': time.strftime('%Y-%m-%d'),
            'usage_based': True,
            'progressive': True
        }

        print(f"    ✅ {pattern_id}: {info['title']}")

    return wear_data

def generate_environmental_damage():
    """Generate environmental damage patterns from field conditions"""

    print("  🌍 Environmental Damage Patterns...")

    environmental_data = {}

    environmental_scenarios = {
        'SALTWATER_CORROSION': {
            'title': 'Saltwater Corrosion Pattern',
            'description': 'Marine environment metal component corrosion',
            'environment_type': 'marine',
            'exposure_duration_days': 90,
            'salt_concentration_ppm': 35000,
            'affected_components': ['mounting_hardware', 'retention_buckles'],
            'corrosion_type': 'pitting_crevice'
        },
        'DESERT_SAND_ABRASION': {
            'title': 'Desert Sand Abrasion Damage',
            'description': 'Silica particle abrasion from desert operations',
            'environment_type': 'arid_desert',
            'particle_size_microns': [50, 200],
            'wind_velocity_mps': 15,
            'exposure_hours': 480,
            'abrasion_pattern': 'windward_surface_erosion'
        },
        'JUNGLE_HUMIDITY_DEGRADATION': {
            'title': 'High Humidity Degradation Pattern',
            'description': 'Tropical environment moisture degradation effects',
            'environment_type': 'tropical_humid',
            'relative_humidity_percent': 95,
            'temperature_c': 35,
            'fungal_growth_risk': 'high',
            'material_effects': ['hydrolysis', 'dimensional_swelling', 'mold_growth']
        }
    }

    for pattern_id, info in environmental_scenarios.items():
        # Model environmental damage progression
        damage_model = model_environmental_damage(info)

        environmental_data[pattern_id] = {
            'id': pattern_id,
            'title': info['title'],
            'source': 'Environmental Exposure Simulation',
            'description': info['description'],
            'defect_type': 'environmental_damage',
            'environmental_scenario': info,
            'damage_model': damage_model,
            'detectability': 'medium_to_high',
            'criticality': 'variable',
            'inspection_method': 'visual_chemical',
            'generated_date': time.strftime('%Y-%m-%d'),
            'environment_specific': True,
            'field_relevant': True
        }

        print(f"    ✅ {pattern_id}: {info['title']}")

    return environmental_data

def calculate_impact_damage_geometry(impact_info: Dict) -> Dict:
    """Calculate detailed impact damage geometry based on physics"""

    if impact_info['impact_type'] == 'ballistic':
        # Ballistic impact creates circular damage with radial cracks
        damage_radius = impact_info.get('damage_radius', 15)

        geometry = {
            'primary_damage': {
                'shape': 'circular',
                'diameter_mm': damage_radius * 2,
                'depth_mm': 2.3,
                'edge_condition': 'fiber_pullout'
            },
            'secondary_damage': {
                'crack_pattern': 'radial_symmetric',
                'crack_count': 6,
                'max_crack_length_mm': damage_radius * 2.5,
                'crack_width_microns': [50, 150]
            },
            'stress_field': {
                'type': 'von_mises',
                'peak_stress_mpa': 450,
                'decay_function': 'exponential',
                'influence_radius_mm': damage_radius * 3
            }
        }

    elif impact_info['impact_type'] == 'blunt_force':
        # Blunt force creates elliptical damage with compression
        contact_area = impact_info.get('contact_area', 1250)

        geometry = {
            'primary_damage': {
                'shape': 'elliptical',
                'major_axis_mm': math.sqrt(contact_area / math.pi) * 2.2,
                'minor_axis_mm': math.sqrt(contact_area / math.pi) * 1.8,
                'depth_mm': 1.5,
                'edge_condition': 'compression_shear'
            },
            'matrix_cracking': {
                'pattern': 'concentric',
                'crack_density': 'high',
                'orientation': 'perpendicular_to_load'
            }
        }

    else:  # Fragment impact
        # Multiple small circular damages
        fragment_count = impact_info.get('fragment_count', 7)

        geometry = {
            'primary_damage': {
                'shape': 'multiple_circular',
                'fragment_count': fragment_count,
                'diameter_range_mm': [3, 12],
                'pattern': 'clustered_random',
                'individual_depth_mm': [0.5, 2.0]
            },
            'interaction_effects': {
                'stress_overlap': True,
                'crack_linking': 'possible',
                'combined_effect': 'additive'
            }
        }

    return geometry

def model_material_degradation(degradation_info: Dict) -> Dict:
    """Model material degradation progression over time"""

    if degradation_info['degradation_type'] == 'photochemical':
        # UV degradation follows molecular chain scission kinetics
        exposure_hours = degradation_info.get('exposure_duration_hours', 2000)

        model = {
            'degradation_mechanism': 'chain_scission',
            'kinetics': {
                'rate_constant': 2.3e-6,  # per hour
                'activation_energy_kj_mol': 120,
                'quantum_efficiency': 0.15
            },
            'property_changes': {
                'tensile_strength_retention_percent': max(70, 100 - exposure_hours * 0.015),
                'impact_strength_retention_percent': max(60, 100 - exposure_hours * 0.020),
                'surface_roughness_increase_ra': exposure_hours * 0.001
            },
            'visual_indicators': {
                'color_change_delta_e': min(8.5, exposure_hours * 0.004),
                'surface_chalking': 'visible' if exposure_hours > 1000 else 'minimal',
                'gloss_reduction_percent': min(40, exposure_hours * 0.018)
            }
        }

    elif degradation_info['degradation_type'] == 'thermal_mechanical':
        # Thermal cycling creates cumulative damage
        cycles = degradation_info.get('cycle_count', 500)

        model = {
            'degradation_mechanism': 'fatigue_creep',
            'damage_accumulation': {
                'damage_per_cycle': 0.002,
                'cumulative_damage': min(1.0, cycles * 0.002),
                'critical_damage': 0.8
            },
            'microstructural_changes': {
                'microcrack_density_per_mm2': cycles * 0.05,
                'crack_length_distribution': 'weibull',
                'interface_debonding_percent': min(25, cycles * 0.05)
            }
        }

    else:  # Chemical degradation
        # Chemical exposure affects polymer matrix
        exposure_hours = degradation_info.get('exposure_duration_hours', 168)

        model = {
            'degradation_mechanism': 'polymer_swelling_extraction',
            'chemical_effects': {
                'mass_change_percent': -2.1,  # plasticizer loss
                'dimensional_change_percent': 0.8,  # swelling then shrinkage
                'modulus_change_percent': -15
            },
            'surface_effects': {
                'etching_depth_microns': exposure_hours * 0.1,
                'surface_energy_change': 'decreased',
                'wettability_change': 'increased_hydrophobic'
            }
        }

    return model

def model_manufacturing_defect(manufacturing_info: Dict) -> Dict:
    """Model manufacturing defect characteristics"""

    if manufacturing_info['defect_type'] == 'void_formation':
        # Model void distribution and characteristics
        void_density = manufacturing_info.get('void_density_per_cm2', 2.3)

        characteristics = {
            'void_statistics': {
                'number_density_per_cm2': void_density,
                'size_distribution': {
                    'type': 'lognormal',
                    'mean_diameter_microns': 150,
                    'standard_deviation': 0.6
                },
                'shape_factor': 0.85,  # sphericity
                'connectivity': 'isolated'
            },
            'formation_physics': {
                'nucleation_sites': 'fiber_matrix_interface',
                'growth_mechanism': 'diffusion_limited',
                'stabilizing_factors': ['surface_tension', 'viscosity']
            },
            'property_impact': {
                'strength_reduction_percent': void_density * 3.2,
                'modulus_reduction_percent': void_density * 1.8,
                'fatigue_life_reduction_percent': void_density * 5.5
            }
        }

    elif manufacturing_info['defect_type'] == 'fiber_orientation':
        # Model fiber misalignment effects
        angle_deviation = abs(manufacturing_info.get('actual_angle_degrees', 15))

        characteristics = {
            'orientation_statistics': {
                'target_angle_degrees': 0,
                'actual_angle_degrees': angle_deviation,
                'standard_deviation_degrees': angle_deviation * 0.3,
                'distribution_type': 'normal'
            },
            'mechanical_impact': {
                'in_plane_strength_reduction_percent': (angle_deviation / 90) ** 2 * 100,
                'shear_strength_change_percent': math.sin(2 * math.radians(angle_deviation)) * 50,
                'stiffness_matrix_rotation': True
            }
        }

    else:  # Delamination
        # Model delamination characteristics
        area = manufacturing_info.get('delamination_area_mm2', 850)

        characteristics = {
            'delamination_geometry': {
                'area_mm2': area,
                'aspect_ratio': 3.2,
                'shape': 'elliptical',
                'edge_condition': 'mixed_mode_crack'
            },
            'interface_properties': {
                'interface_strength_mpa': 45,
                'fracture_toughness_j_m2': 250,
                'mixed_mode_ratio': 0.6
            },
            'propagation_tendency': {
                'critical_load_n': 1200,
                'growth_rate_mm_cycle': 0.003,
                'paris_law_constants': {'C': 2.1e-8, 'm': 3.4}
            }
        }

    return characteristics

def model_wear_progression(wear_info: Dict) -> Dict:
    """Model wear pattern progression over usage cycles"""

    wear_type = wear_info.get('wear_type', 'abrasive_contact')
    cycles = wear_info.get('use_cycles', 200)

    if wear_type == 'abrasive_contact':
        # Archard wear equation application
        progression = {
            'wear_mechanism': 'abrasive_adhesive',
            'wear_equation': 'archard',
            'parameters': {
                'wear_coefficient': 1.2e-6,
                'hardness_mpa': 180,
                'contact_pressure_mpa': wear_info.get('contact_pressure_kpa', 15) / 1000
            },
            'progression_model': {
                'wear_volume_mm3': cycles * 0.015,
                'wear_depth_microns': cycles * 0.225,
                'surface_roughness_ra': 1.2 + cycles * 0.008
            },
            'geometry_evolution': {
                'contact_area_increase_percent': cycles * 0.12,
                'stress_redistribution': True,
                'shape_factor_change': -0.05  # becomes more conformal
            }
        }

    elif wear_type == 'adhesive_abrasive':
        # Metal-on-composite wear
        progression = {
            'wear_mechanism': 'adhesive_with_abrasive_particles',
            'material_transfer': {
                'aluminum_transfer_to_composite': True,
                'composite_debris_formation': True,
                'third_body_abrasion': 'significant'
            },
            'groove_formation': {
                'groove_depth_mm': cycles * 0.002,
                'groove_width_mm': cycles * 0.003,
                'groove_spacing_mm': 0.8
            }
        }

    else:  # Viscoelastic deformation
        # Foam compression set
        progression = {
            'deformation_mechanism': 'viscoelastic_creep',
            'time_dependent_response': {
                'instantaneous_compression_percent': 60,
                'delayed_recovery_hours': 24,
                'permanent_set_percent': min(20, cycles * 0.03)
            },
            'material_property_changes': {
                'elastic_modulus_reduction_percent': cycles * 0.05,
                'energy_absorption_reduction_percent': cycles * 0.08
            }
        }

    return progression

def model_environmental_damage(environmental_info: Dict) -> Dict:
    """Model environmental damage progression"""

    environment_type = environmental_info.get('environment_type', 'marine')
    exposure_days = environmental_info.get('exposure_duration_days', 90)

    if environment_type == 'marine':
        # Saltwater corrosion modeling
        model = {
            'corrosion_mechanism': 'electrochemical',
            'corrosion_rate_mm_year': 0.15,
            'pit_formation': {
                'pit_density_per_cm2': exposure_days * 0.02,
                'average_pit_depth_microns': exposure_days * 1.2,
                'pit_diameter_microns': exposure_days * 0.8
            },
            'material_effects': {
                'mass_loss_percent': exposure_days * 0.001,
                'strength_retention_percent': max(85, 100 - exposure_days * 0.05)
            }
        }

    elif environment_type == 'arid_desert':
        # Sand abrasion modeling
        exposure_hours = environmental_info.get('exposure_hours', 480)

        model = {
            'abrasion_mechanism': 'particle_impact_erosion',
            'erosion_rate': {
                'mass_loss_g_h': 0.003,
                'surface_recession_microns_h': 0.12,
                'roughness_increase_ra_h': 0.008
            },
            'particle_effects': {
                'embedded_particles': True,
                'surface_texture_change': 'matte_finish',
                'optical_property_change': 'reduced_reflectance'
            }
        }

    else:  # Tropical humid
        # High humidity degradation
        humidity_percent = environmental_info.get('relative_humidity_percent', 95)

        model = {
            'degradation_mechanism': 'hydrolytic_biological',
            'moisture_effects': {
                'moisture_uptake_percent': 1.8,
                'dimensional_swelling_percent': 0.3,
                'tg_depression_c': -12
            },
            'biological_effects': {
                'fungal_growth_probability': 0.85,
                'growth_rate_mm_day': 0.15,
                'biodeterioration_severity': 'moderate'
            }
        }

    return model

def save_defect_patterns(defect_data: Dict):
    """Save synthetic defect patterns to JSON files"""

    print(f"\n💾 Saving {len(defect_data)} defect patterns to {BASE_DIR}...")

    os.makedirs(BASE_DIR, exist_ok=True)

    # Save each defect pattern as individual JSON file
    for pattern_id, data in defect_data.items():
        filename = f"{pattern_id.replace(':', '_').replace('-', '_').replace('/', '_')}.json"
        filepath = os.path.join(BASE_DIR, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  ✅ Saved: {filename}")

    # Save combined defect pattern database
    combined_file = os.path.join(BASE_DIR, 'defect_patterns_database.json')
    with open(combined_file, 'w', encoding='utf-8') as f:
        json.dump(defect_data, f, indent=2, ensure_ascii=False)

    print(f"  ✅ Saved combined database: defect_patterns_database.json")

def main():
    """Main defect pattern generation function"""

    print("🚀 Starting Physics-Based Synthetic Defect Pattern Generation\n")
    print("🔬 Generating realistic defect patterns based on materials science...")
    print("💥 Impact damage, material degradation, manufacturing defects, wear patterns\n")

    try:
        # Generate synthetic defect patterns
        defect_data = generate_synthetic_defects()

        if defect_data:
            # Save pattern data
            save_defect_patterns(defect_data)

            print(f"\n✅ Successfully generated {len(defect_data)} synthetic defect patterns!")
            print(f"📊 Patterns saved to: {BASE_DIR}")

            # Show what we generated
            print("\n📋 Generated Defect Categories:")
            categories = {}
            for pattern_id, data in defect_data.items():
                category = data.get('defect_type', 'unknown')
                if category not in categories:
                    categories[category] = 0
                categories[category] += 1

            for category, count in categories.items():
                print(f"  • {category.replace('_', ' ').title()}: {count} patterns")

            print("\n🔬 Defect Pattern Summary:")
            for pattern_id, data in defect_data.items():
                criticality = data.get('criticality', 'unknown')
                detectability = data.get('detectability', 'unknown')
                print(f"  • {pattern_id}: {data.get('title', 'No title')}")
                print(f"    Criticality: {criticality} | Detectability: {detectability}")

        else:
            print("\n⚠️ No defect patterns could be generated at this time.")

    except Exception as e:
        print(f"\n❌ Pattern generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    main()