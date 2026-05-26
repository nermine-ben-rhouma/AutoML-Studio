import { useState } from "react";
import { login, register } from "../api";
import { setAuth } from "../authStorage";

export default function LoginPage({ onSuccess, defaultEmail }) {
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState(defaultEmail || "");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      const data =
        mode === "login"
          ? await login(email, password)
          : await register(email, password, fullName);
      setAuth(data.access_token, data.user);
      onSuccess(data.user);
    } catch (err) {
      setError(err.message || "Erreur de connexion");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-card">
        <div className="auth-logo">⚗️</div>
        <h1>AutoML Studio</h1>
        <p className="auth-sub">Connectez-vous pour accéder à la plateforme ML</p>

        <div className="auth-tabs">
          <button
            type="button"
            className={mode === "login" ? "auth-tab active" : "auth-tab"}
            onClick={() => setMode("login")}
          >
            Connexion
          </button>
          <button
            type="button"
            className={mode === "register" ? "auth-tab active" : "auth-tab"}
            onClick={() => setMode("register")}
          >
            Inscription
          </button>
        </div>

        <form onSubmit={submit} className="auth-form">
          {mode === "register" && (
            <label>
              Nom complet
              <input
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Votre nom"
                autoComplete="name"
              />
            </label>
          )}
          <label>
            Email
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="vous@exemple.com"
              required
              autoComplete="email"
            />
          </label>
          <label>
            Mot de passe
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={mode === "register" ? "6 caractères minimum" : "••••••••"}
              required
              minLength={mode === "register" ? 6 : 1}
              autoComplete={mode === "login" ? "current-password" : "new-password"}
            />
          </label>

          {error && <p className="auth-error">{error}</p>}

          <button type="submit" className="auth-submit" disabled={loading}>
            {loading ? "Chargement…" : mode === "login" ? "Se connecter" : "Créer un compte"}
          </button>
        </form>

        {defaultEmail && mode === "login" && (
          <p className="auth-hint">
            Compte par défaut au premier lancement : <strong>{defaultEmail}</strong> /{" "}
            <strong>admin123</strong> (changez le mot de passe en production)
          </p>
        )}
      </div>
    </div>
  );
}
