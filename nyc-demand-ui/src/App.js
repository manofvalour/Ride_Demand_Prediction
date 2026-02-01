import React, { useEffect, useState } from 'react';
import { MapContainer, TileLayer, GeoJSON } from 'react-leaflet';
import { Activity, CloudRain, Thermometer, Navigation, AlertCircle } from 'lucide-react';
import 'leaflet/dist/leaflet.css';

const NYC_GEOJSON_URL = "/NYC_Taxi_Zones.geojson";
const API_URL = "https://special-spork-j9q69qr9wxjh7q6-8000.app.github.dev/api/demand";

function App() {
  const [predictionData, setPredictionData] = useState(null);
  const [geoJson, setGeoJson] = useState(null);
  const [selectedZone, setSelectedZone] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log("Connecting to API...");
        const res = await fetch(API_URL);
        
        if (!res.ok) throw new Error(`Backend Unreachable (Status: ${res.status}). Ensure Port 8000 is set to Public!`);
        
        const data = await res.json();
        console.log("Data received:", data);

        const geoRes = await fetch(NYC_GEOJSON_URL);
        const geoData = await geoRes.json();
        
        setPredictionData(data);
        setGeoJson(geoData);
      } catch (err) {
        console.error("Fetch error:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const getStyle = (feature) => {
    // Force ID to string to match JSON keys
    const zoneId = String(feature.properties.location_id);
    const zoneInfo = predictionData?.predictions?.[zoneId];
    const demand = zoneInfo?.target_hvfhv || 0;

    return {
      fillColor: demand > 100 ? '#4a148c' : 
                 demand > 50  ? '#7b1fa2' : 
                 demand > 10  ? '#9c27b0' : '#f3e5f5',
      weight: 0.5,
      opacity: 1,
      color: 'white',
      fillOpacity: 0.7
    };
  };

  if (loading) {
    return (
      <div className="flex flex-col h-screen items-center justify-center bg-slate-900 text-white">
        <div className="animate-spin mb-4 text-blue-400"><Navigation size={40} /></div>
        <p className="text-lg font-medium animate-pulse">Syncing with NYC Traffic Data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col h-screen items-center justify-center bg-slate-50 p-6 text-center">
        <AlertCircle size={50} className="text-red-500 mb-4" />
        <h2 className="text-2xl font-bold text-slate-800">Connection Error</h2>
        <p className="text-slate-600 mt-2 max-w-md">{error}</p>
        <button 
          onClick={() => window.location.reload()}
          className="mt-6 px-6 py-2 bg-blue-600 text-white rounded-full font-bold hover:bg-blue-700 transition"
        >
          Retry Connection
        </button>
      </div>
    );
  }

  // Ensure currentZoneInfo is accessed safely using a string key
  const currentZoneInfo = selectedZone ? predictionData?.predictions?.[String(selectedZone)] : null;

  return (
    <div className="flex flex-col h-screen bg-slate-50 font-sans">
      <header className="bg-slate-900 text-white p-4 shadow-lg flex justify-between items-center z-[1000]">
        <h1 className="text-xl font-bold flex items-center gap-2">
          <Navigation size={24} className="text-blue-400" /> NYC Demand Intelligence
        </h1>
        <div className="text-right">
          <p className="text-[10px] text-slate-400 uppercase tracking-widest font-bold">Live API Status</p>
          <p className="text-xs text-green-400 flex items-center gap-1 justify-end">
            <span className="w-2 h-2 bg-green-400 rounded-full animate-ping"></span> Connected
          </p>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">
        {/* Map Section */}
        <div className="w-2/3 h-full relative border-r border-slate-200">
          <MapContainer center={[40.7128, -74.0060]} zoom={11} className="h-full w-full">
            <TileLayer url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png" />
            {geoJson && (
              <GeoJSON 
                data={geoJson} 
                style={getStyle}
                onEachFeature={(feature, layer) => {
                  layer.on('click', () => setSelectedZone(feature.properties.location_id));
                  layer.bindTooltip(`${feature.properties.zone}`, { sticky: true });
                }}
              />
            )}
          </MapContainer>
        </div>

        {/* Info Sidebar */}
        <div className="w-1/3 p-6 overflow-y-auto bg-white shadow-inner">
          {!selectedZone ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-400 text-center space-y-4">
              <Activity size={48} className="opacity-20" />
              <p className="text-sm">Click any NYC taxi zone on the map<br/>to view demand predictions.</p>
            </div>
          ) : (
            <div className="space-y-6 animate-in fade-in slide-in-from-right-4 duration-300">
              <div>
                <p className="text-xs font-bold text-blue-600 uppercase">Selected Neighborhood</p>
                <h2 className="text-2xl font-black text-slate-800">Zone ID: {selectedZone}</h2>
              </div>
              
              <div className="grid grid-cols-1 gap-4">
                <StatCard label="Uber/Lyft Demand" value={currentZoneInfo?.target_hvfhv} color="text-purple-600" icon={<Activity />} />
                <StatCard label="Yellow Taxi" value={currentZoneInfo?.target_yellow} color="text-yellow-600" icon={<Activity />} />
              </div>

              <div className="bg-slate-50 p-5 rounded-2xl border border-slate-100 space-y-4">
                <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-tighter">Environment Metrics</h3>
                <div className="flex justify-between items-center">
                  <span className="flex items-center gap-2 text-slate-600 font-medium"><Thermometer size={16} className="text-orange-500"/> Temperature</span>
                  <span className="font-bold text-slate-800">{currentZoneInfo?.feelslike || '--'}°F</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="flex items-center gap-2 text-slate-600 font-medium"><CloudRain size={16} className="text-blue-500"/> Precipitation</span>
                  <span className="font-bold text-slate-800">{currentZoneInfo?.precip || '0'} in</span>
                </div>
                <div className="pt-4 border-t border-slate-200">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-slate-500 font-bold">CONGESTION INDEX</span>
                    <span className="text-blue-600 font-bold">{((currentZoneInfo?.zone_congestion_index || 0) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden">
                    <div 
                      className="bg-blue-600 h-full transition-all duration-500" 
                      style={{ width: `${(currentZoneInfo?.zone_congestion_index || 0) * 100}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

const StatCard = ({ label, value, color, icon }) => (
  <div className="p-4 rounded-2xl border border-slate-100 bg-white shadow-sm flex items-center justify-between">
    <div>
      <p className="text-[10px] text-slate-400 font-bold uppercase">{label}</p>
      <p className={`text-3xl font-black ${color}`}>{Math.round(value) || 0}</p>
    </div>
    <div className={`${color} opacity-20`}>{icon}</div>
  </div>
);

export default App;