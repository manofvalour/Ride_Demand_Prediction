// src/components/NYCMapApp.jsx
import React, { useState, useEffect, useMemo, useRef } from 'react';
import { MapContainer, TileLayer, GeoJSON, useMap } from 'react-leaflet';
import { Search, Moon, Sun, MapPin } from 'lucide-react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import styles from './App.css';

const API_URL = "http://127.0.0.1:8000/api/demand";
const GEOJSON_FILE = "/taxi_zones.json"; 
const THRESHOLDS = { LOW: 0.0270, MEDIUM: 0.0421 };

// Helper to handle map movements
const MapController = ({ selectedZoneId, geoData, boroughFilter }) => {
    const map = useMap();
    
    useEffect(() => {
        if (!map || !geoData) return;
        
        // If a specific zone is selected, zoom to it
        if (selectedZoneId) {
            const feature = geoData.features.find(f => (f.properties.location_id || f.properties.OBJECTID) == selectedZoneId);
            if (feature) {
                const bounds = L.geoJSON(feature).getBounds();
                map.flyToBounds(bounds, { padding: [100, 100], maxZoom: 14 });
            }
        } 
    }, [selectedZoneId, map, geoData]);

    return null;
};

const NYCMapApp = () => {
    const [taxiData, setTaxiData] = useState({});
    const [geoData, setGeoData] = useState(null);
    const [selectedZoneId, setSelectedZoneId] = useState(null);
    const [theme, setTheme] = useState('light');
    const [searchQuery, setSearchQuery] = useState('');
    const [boroughFilter, setBoroughFilter] = useState('All');
    const [time, setTime] = useState(new Date());

    useEffect(() => {
        const timer = setInterval(() => setTime(new Date()), 1000);
        const loadData = async () => {
            try {
                const [dRes, gRes] = await Promise.all([
                    fetch(API_URL).then(r => r.json()),
                    fetch(GEOJSON_FILE).then(r => r.json())
                ]);
                setTaxiData(dRes.predictions || {});
                setGeoData(gRes);
            } catch (e) { console.error("Fetch error:", e); }
        };
        loadData();
        return () => clearInterval(timer);
    }, []);

    const helpers = {
        getCongestion: (val) => {
            if (!val || val <= THRESHOLDS.LOW) return { label: 'Low', class: 'status-low', color: '#166534', bg: '#dcfce7' };
            if (val <= THRESHOLDS.MEDIUM) return { label: 'Med', class: 'status-medium', color: '#92400e', bg: '#fef3c7' };
            return { label: 'High', class: 'status-high', color: '#991b1b', bg: '#fee2e2' };
        },
        getColor: (d) => d > 200 ? '#312e81' : d > 120 ? '#4338ca' : d > 60 ? '#6366f1' : d > 20 ? '#a5b4fc' : d > 0 ? '#e0e7ff' : '#f1f5f9',
        getSpeed: (idx) => Math.max(5, (25 - (idx * 200))).toFixed(1)
    };

    const displayList = useMemo(() => {
        if (!geoData) return [];
        return geoData.features
            .filter(f => {
                const matchesB = boroughFilter === 'All' || f.properties.borough === boroughFilter;
                const matchesS = f.properties.zone.toLowerCase().includes(searchQuery.toLowerCase());
                return matchesB && matchesS;
            })
            .map(f => {
                const id = f.properties.location_id || f.properties.OBJECTID;
                const stats = taxiData[id] || { target_yellow: 0, target_green: 0, target_hvfhv: 0, zone_congestion_index: 0 };
                return { f, id, stats, total: stats.target_yellow + stats.target_green + stats.target_hvfhv };
            })
            .sort((a, b) => b.total - a.total)
            .slice(0, 20);
    }, [geoData, taxiData, boroughFilter, searchQuery]);

    const city = Object.values(taxiData)[0] || {};

    return (
        <div className={`${styles.appLayout} ${theme}`}>
            <header className={styles.header}>
                <div className={styles.clockSection}>
                    <div className={styles.liveClock}>{time.toLocaleTimeString()}</div>
                    <div className={styles.liveDate}>{time.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })}</div>
                </div>
                <div className={styles.counter}>
                    <span className={styles.label}>Avg Speed</span>
                    <span className={styles.value}>{city.city_congestion_index ? helpers.getSpeed(city.city_congestion_index) : '--'} MPH</span>
                </div>
                <div className={styles.counter}>
                    <span className={styles.label}>Congestion</span>
                    <span className={styles.value}>{city.city_congestion_index?.toFixed(3) || '--'}</span>
                </div>
                <button className={styles.themeToggle} onClick={() => setTheme(t => t === 'light' ? 'dark' : 'light')}>
                    {theme === 'light' ? <Moon size={18} /> : <Sun size={18} />}
                </button>
                <div style={{ marginLeft: 'auto', background: '#fee2e2', color: '#b91c1c', padding: '6px 12px', borderRadius: '6px', fontSize: '0.6rem', fontWeight: 900 }}>
                    <MapPin size={12} style={{ verticalAlign: 'middle', marginRight: '4px' }} /> LIVE FEED
                </div>
            </header>

            <div className={styles.mainContainer}>
                <aside className={styles.sidebar}>
                    <div className={styles.controls}>
                        <div className={styles.searchWrapper}>
                            <Search size={16} />
                            <input placeholder="Search Zone..." value={searchQuery} onChange={e => setSearchQuery(e.target.value)} />
                        </div>
                        <select value={boroughFilter} onChange={e => setBoroughFilter(e.target.value)}>
                            <option value="All">All Boroughs</option>
                            <option value="Manhattan">Manhattan</option>
                            <option value="Brooklyn">Brooklyn</option>
                            <option value="Queens">Queens</option>
                            <option value="Bronx">Bronx</option>
                            <option value="Staten Island">Staten Island</option>
                        </select>
                    </div>
                    <div className={styles.scrollArea}>
                        {displayList.map(item => {
                            const cong = helpers.getCongestion(item.stats.zone_congestion_index);
                            return (
                                <div key={item.id} className={`${styles.card} ${selectedZoneId === item.id ? styles.activeCard : ''}`} onClick={() => setSelectedZoneId(item.id)}>
                                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                        <div>
                                            <div style={{ fontWeight: 800, fontSize: '0.8rem' }}>{item.f.properties.zone}</div>
                                            <div style={{ fontSize: '0.6rem', background: cong.bg, color: cong.color, display: 'inline-block', padding: '2px 5px', borderRadius: '4px', marginTop: '4px' }}>
                                                {cong.label} Traffic
                                            </div>
                                        </div>
                                        <div style={{ fontWeight: 900, fontSize: '0.7rem' }}>{item.stats.zone_congestion_index.toFixed(3)}</div>
                                    </div>
                                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '5px', marginTop: '10px' }}>
                                        <div className={styles.statBox} style={{ background: '#fef9c3', color: '#854d0e' }}>Y: {Math.round(item.stats.target_yellow)}</div>
                                        <div className={styles.statBox} style={{ background: '#dcfce7', color: '#166534' }}>G: {Math.round(item.stats.target_green)}</div>
                                        <div className={styles.statBox} style={{ background: '#ede9fe', color: '#5b21b6' }}>F: {Math.round(item.stats.target_hvfhv)}</div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </aside>

                <div className={styles.mapWrapper}>
                    <MapContainer center={[40.7128, -74.0060]} zoom={11} zoomControl={false} style={{ height: '100%', width: '100%' }}>
                        <TileLayer url={theme === 'dark' ? 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png' : 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png'} />
                        {geoData && <GeoJSON 
                            data={geoData} 
                            style={(f) => {
                                const id = f.properties.location_id || f.properties.OBJECTID;
                                const d = taxiData[id] || {};
                                return { fillColor: helpers.getColor(d.target_yellow + d.target_green + d.target_hvfhv || 0), fillOpacity: 0.7, weight: 1, color: 'white' };
                            }}
                            onEachFeature={(f, layer) => {
                                layer.on('click', () => setSelectedZoneId(f.properties.location_id || f.properties.OBJECTID));
                            }}
                        />}
                        <MapController selectedZoneId={selectedZoneId} geoData={geoData} />
                    </MapContainer>
                </div>
            </div>
        </div>
    );
};

export default NYCMapApp;