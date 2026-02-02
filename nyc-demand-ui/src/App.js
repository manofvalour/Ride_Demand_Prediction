import React, { useEffect, useState } from 'react';
import {
  Activity,
  AlertCircle,
  CloudRain,
  Navigation,
  Search,
  Thermometer,
  TrendingUp,
  X,
} from 'lucide-react';

const API_URL = "https://special-spork-j9q69qr9wxjh7q6-8000.app.github.dev/api/demand";

function App() {
  const [data, setData] = useState(null);
  const [selectedZoneId, setSelectedZoneId] = useState(null);
  const [search, setSearch] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch(API_URL)
      .then(r => {
        if (!r.ok) throw new Error("Backend unreachable (port 8000 public?)");
        return r.json();
      })
      .then(json => setData(json.predictions))
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const zones = data ? Object.entries(data) : [];
  const filtered = zones.filter(([id]) => id.toLowerCase().includes(search.toLowerCase().trim()));

  const selected = selectedZoneId ? data[selectedZoneId] : null;

  if (loading) return <LoadingView />;
  if (error) return <ErrorView message={error} />;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 text-slate-900 antialiased flex flex-col">
      {/* Header + KPI strip */}
      <header className="bg-white border-b border-slate-200/80 shadow-sm">
        <div className="max-w-screen-2xl mx-auto px-5 sm:px-6 lg:px-8">
          {/* Brand + Search */}
          <div className="flex items-center justify-between py-5 gap-6">
            <div className="flex items-center gap-3.5">
              <div className="bg-gradient-to-br from-indigo-600 to-indigo-700 text-white p-2.5 rounded-xl shadow-md">
                <Navigation size={22} />
              </div>
              <h1 className="text-2xl font-extrabold tracking-tight text-slate-900">
                NYC Demand
              </h1>
            </div>

            <div className="relative max-w-md w-full">
              <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-400" size={18} />
              <input
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search zone ID..."
                className="
                  w-full pl-12 pr-5 py-3 bg-white border border-slate-200 rounded-xl
                  text-sm placeholder-slate-400
                  focus:border-indigo-400 focus:ring-2 focus:ring-indigo-400/20
                  outline-none transition-all duration-200 shadow-sm
                "
              />
            </div>
          </div>

          {/* KPI Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-5 pb-8">
            <KpiCard
              icon={<Activity size={18} />}
              label="Zones"
              value={filtered.length}
              trend="+4"
              trendColor="text-emerald-600"
            />
            <KpiCard
              icon={<TrendingUp size={18} />}
              label="High Demand"
              value={filtered.filter(([,v]) => Math.round(v.target_hvfhv) >= 80).length}
              trend="+7"
              trendColor="text-emerald-600"
            />
            <KpiCard
              icon={<Navigation size={18} />}
              label="Avg HVFHV"
              value={Math.round(filtered.reduce((s,[,v])=>s+v.target_hvfhv,0) / (filtered.length || 1))}
              trend="–1.8%"
              trendColor="text-amber-600"
            />
            <KpiCard
              icon={<CloudRain size={18} />}
              label="Rain Impact"
              value={filtered.filter(([,v]) => v.precip > 0).length}
              trend="live"
              trendColor="text-slate-500"
            />
          </div>
        </div>
      </header>

      {/* Main layout */}
      <div className="flex flex-1 overflow-hidden relative">
        {/* Left navigation sidebar */}
        <aside className="hidden lg:block w-72 bg-white/95 backdrop-blur-sm border-r border-slate-200/70 overflow-y-auto">
          <NavSidebar />
        </aside>

        {/* Zone grid */}
        <main className="flex-1 overflow-y-auto p-6 lg:p-8">
          <div className="max-w-screen-2xl mx-auto">
            <h2 className="text-2xl font-bold text-slate-900 mb-7">
              Zone Forecasts
              <span className="ml-3 text-base font-normal text-slate-500">
                {filtered.length} active zones
              </span>
            </h2>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-6">
              {filtered.map(([id, stats]) => (
                <ZoneCard
                  key={id}
                  id={id}
                  stats={stats}
                  isActive={selectedZoneId === id}
                  onSelect={() => setSelectedZoneId(id)}
                />
              ))}
            </div>
          </div>
        </main>

        {/* Detail panel */}
        <div
          className={`
            fixed lg:sticky lg:top-0 inset-y-0 right-0 z-40 w-full lg:w-[420px]
            bg-white/98 backdrop-blur-xl border-l border-slate-200/70 shadow-2xl lg:shadow-none
            transform transition-transform duration-400 ease-in-out
            ${selectedZoneId ? 'translate-x-0' : 'translate-x-full lg:translate-x-0'}
          `}
        >
          {selected && (
            <ZoneDetail
              zoneId={selectedZoneId}
              data={selected}
              onClose={() => setSelectedZoneId(null)}
            />
          )}
        </div>
      </div>
    </div>
  );
}

/* ── Components ──────────────────────────────────────────────────── */

function KpiCard({ icon, label, value, trend, trendColor }) {
  return (
    <div className="
      bg-white rounded-2xl border border-slate-200/70 shadow-sm
      hover:shadow-md hover:border-slate-300 transition-all duration-300 p-6
    ">
      <div className="flex items-center gap-3 mb-4 text-slate-600">
        {icon}
        <span className="text-sm font-medium">{label}</span>
      </div>
      <div className="text-3xl font-extrabold tracking-tight text-slate-900">
        {value}
      </div>
      <div className={`text-sm font-semibold mt-2 ${trendColor}`}>{trend}</div>
    </div>
  );
}

function ZoneCard({ id, stats, isActive, onSelect }) {
  const hv = Math.round(stats.target_hvfhv);

  const getStyle = (val) => {
    if (val >= 120) return {
      bg: 'bg-gradient-to-br from-emerald-50 to-emerald-100/40',
      border: 'border-emerald-200/70',
      text: 'text-emerald-800',
      ring: 'ring-emerald-300/30'
    };
    if (val >= 70) return {
      bg: 'bg-gradient-to-br from-blue-50 to-blue-100/40',
      border: 'border-blue-200/70',
      text: 'text-blue-800',
      ring: 'ring-blue-300/30'
    };
    if (val >= 30) return {
      bg: 'bg-gradient-to-br from-amber-50 to-amber-100/40',
      border: 'border-amber-200/70',
      text: 'text-amber-800',
      ring: 'ring-amber-300/30'
    };
    return {
      bg: 'bg-white',
      border: 'border-slate-200/70',
      text: 'text-slate-900',
      ring: ''
    };
  };

  const style = getStyle(hv);

  return (
    <div
      onClick={onSelect}
      className={`
        group relative overflow-hidden rounded-2xl border transition-all duration-300 ease-out
        hover:shadow-xl hover:-translate-y-1 hover:border-indigo-200/60
        ${isActive 
          ? 'ring-2 ring-offset-2 ring-indigo-400/40 shadow-lg scale-[1.015]' 
          : style.border
        }
        ${style.bg}
        cursor-pointer
      `}
    >
      {/* Subtle hover overlay */}
      <div className="absolute inset-0 bg-gradient-to-br from-white/30 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />

      <div className="p-6 relative">
        <div className="flex items-start justify-between mb-4">
          <div className="text-xs font-semibold px-3 py-1.5 rounded-full bg-white/70 backdrop-blur-sm border border-slate-200/60 shadow-sm">
            Zone {id}
          </div>
          <TrendingUp 
            size={20} 
            className={`${hv >= 70 ? 'text-emerald-500' : 'text-slate-400'} transition-colors group-hover:scale-110 duration-300`} 
          />
        </div>

        <div className="mb-6">
          <div className="text-sm text-slate-600 mb-1.5 font-medium">High-volume for-hire</div>
          <div className={`text-4xl font-extrabold tracking-tight ${style.text}`}>
            {hv}
            <span className="text-xl font-semibold text-slate-400 ml-2">trips</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 text-sm pt-4 border-t border-slate-200/50">
          <div>
            <div className="text-slate-500 text-xs">Yellow</div>
            <div className="font-semibold text-yellow-700 mt-0.5">{Math.round(stats.target_yellow)}</div>
          </div>
          <div>
            <div className="text-slate-500 text-xs">Green</div>
            <div className="font-semibold text-green-700 mt-0.5">{Math.round(stats.target_green || 0)}</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function ZoneDetail({ zoneId, data, onClose }) {
  return (
    <div className="h-full flex flex-col">
      <div className="p-6 border-b border-slate-200/80 flex items-center justify-between">
        <h2 className="text-2xl font-bold tracking-tight text-slate-900">
          Zone {zoneId}
        </h2>
        <button
          onClick={onClose}
          className="p-2.5 rounded-full hover:bg-slate-100 transition-colors lg:hidden"
        >
          <X size={20} className="text-slate-600" />
        </button>
      </div>

      <div className="p-6 flex-1 overflow-y-auto space-y-10">
        {/* Weather */}
        <div className="grid grid-cols-2 gap-6">
          <div className="bg-gradient-to-br from-orange-50 to-orange-100/30 p-6 rounded-2xl border border-orange-200/50 shadow-sm hover:shadow-md transition-all">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-orange-100/60 p-2.5 rounded-lg">
                <Thermometer className="text-orange-600" size={20} />
              </div>
              <div className="text-sm font-medium text-orange-800">Feels like</div>
            </div>
            <div className="text-3xl font-extrabold text-orange-900">
              {data.feelslike}°F
            </div>
          </div>

          <div className="bg-gradient-to-br from-blue-50 to-blue-100/30 p-6 rounded-2xl border border-blue-200/50 shadow-sm hover:shadow-md transition-all">
            <div className="flex items-center gap-3 mb-4">
              <div className="bg-blue-100/60 p-2.5 rounded-lg">
                <CloudRain className="text-blue-600" size={20} />
              </div>
              <div className="text-sm font-medium text-blue-800">Precipitation</div>
            </div>
            <div className="text-3xl font-extrabold text-blue-900">
              {data.precip}"
            </div>
          </div>
        </div>

        {/* Demand breakdown */}
        <div className="space-y-7">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-500">
            Demand Breakdown
          </h3>

          <DemandBar label="High-volume (Uber/Lyft)" value={data.target_hvfhv} color="bg-indigo-500" />
          <DemandBar label="Yellow medallion" value={data.target_yellow} color="bg-yellow-500" />
          <DemandBar label="Green street hail" value={data.target_green || 0} color="bg-emerald-500" />
        </div>
      </div>
    </div>
  );
}

function DemandBar({ label, value, color }) {
  const pct = Math.min(Math.round(value) / 2.5, 100);
  return (
    <div>
      <div className="flex justify-between items-baseline text-sm mb-2">
        <span className="font-medium text-slate-700">{label}</span>
        <span className="font-bold text-slate-900">{Math.round(value)}</span>
      </div>
      <div className="h-3 bg-slate-100 rounded-full overflow-hidden ring-1 ring-inset ring-slate-200/50">
        <div
          className={`${color} h-full rounded-full transition-all duration-1000 ease-out`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

function NavSidebar() {
  return (
    <div className="py-8 px-5">
      <div className="flex items-center gap-3 mb-8 px-2">
        <div className="bg-indigo-100/60 p-2.5 rounded-lg">
          <Navigation size={20} className="text-indigo-700" />
        </div>
        <h3 className="text-lg font-semibold text-slate-800">Controls</h3>
      </div>

      <nav className="space-y-1.5">
        {['Dashboard', 'All Zones', 'Forecast Trends', 'Weather Impact', 'Settings'].map(item => (
          <a
            key={item}
            href="#"
            className={`
              block px-4 py-3 rounded-xl text-sm font-medium transition-colors
              ${item === 'Dashboard'
                ? 'bg-indigo-50 text-indigo-700 font-semibold'
                : 'text-slate-600 hover:bg-slate-50 hover:text-slate-800'}
            `}
          >
            {item}
          </a>
        ))}
      </nav>
    </div>
  );
}

function LoadingView() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-white flex items-center justify-center">
      <div className="text-center space-y-5">
        <div className="w-16 h-16 rounded-full border-4 border-indigo-200 border-t-indigo-600 animate-spin mx-auto" />
        <p className="text-lg font-medium text-slate-700 tracking-wide">
          Loading real-time demand...
        </p>
      </div>
    </div>
  );
}

function ErrorView({ message }) {
  return (
    <div className="min-h-screen bg-slate-50 flex items-center justify-center p-6">
      <div className="max-w-md text-center space-y-6">
        <AlertCircle size={72} className="text-red-500 mx-auto" />
        <h2 className="text-2xl font-bold text-slate-800">Connection Failed</h2>
        <p className="text-slate-600 leading-relaxed">{message}</p>
        <button
          onClick={() => window.location.reload()}
          className="
            px-10 py-4 bg-slate-900 text-white rounded-xl font-medium
            hover:bg-slate-800 transition-all duration-200 shadow-lg hover:shadow-xl
          "
        >
          Try Again
        </button>
      </div>
    </div>
  );
}

export default App;